import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import CIFARDataLoader
from ddpm_loss import SimplifiedDDPMLoss
from forward_process import ForwardDiffusion
from tqdm import tqdm

"""
1. Build a simple noise prediction network ε_θ(x_t, t)
2. Understand time step embedding (sinusoidal encoding)
3. Train the network on CIFAR-10
4. Visualize what the network learns

"""


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time step embedding

    Formula:
        PE(t, 2i) = sin(t / 10000^(2i/d))
        PE(t, 2i+1) = cos(t / 10000^(2i/d))

    where d is the embedding dimension.
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        """
        Args:
            timesteps: [B] tensor of timesteps (0 to T-1)
        Returns:
            embeddings: [B, embedding_dim] tensor
        """

        device = timesteps.device
        half_dim = self.embedding_dim // 2

        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)

        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        if self.embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


class ResidualBlock(nn.Module):
    """
    Residual block with time embedding injection.

    Architecture:
        x --> [GN -> SiLU -> Conv] --> [+ time_emb] --> [GN -> SiLU -> Conv] --> + x --> out
    """

    def __init__(self, in_channels, out_channels, time_embed_dim):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Linear(time_embed_dim, out_channels)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.act = nn.SiLU()

        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()

    def forward(self, x, t_emb):
        """
        Args:
            x: [B, C, H, W]
            t_emb: [B, time_embed_dim]
        """
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        time_bias = self.time_proj(self.act(t_emb))
        h = h + time_bias.view(time_bias.shape[0], time_bias.shape[1], 1, 1)

        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)

        return h + self.residual(x)


class SimpleUNet(nn.Module):
    """
    A simplified U-Net for noise prediction

    Architecture:
    - Encoder: Downsample with conv layers
    - Middle: Process at lowest resolution
    - Decoder: Upsample with transposed conv
    - Skip connections: Concatenate encoder features to decoder
    """

    def __init__(
        self, in_channels=3, out_channels=3, base_channels=64, time_embed_dim=128
    ):
        super().__init__()

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(embedding_dim=time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

        # Initial projection
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder
        self.enc1 = self._make_layer(base_channels, base_channels, time_embed_dim)
        self.down1 = nn.Conv2d(
            base_channels, base_channels, kernel_size=3, stride=2, padding=1
        )

        self.enc2 = self._make_layer(base_channels, base_channels * 2, time_embed_dim)
        self.down2 = nn.Conv2d(
            base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1
        )

        self.enc3 = self._make_layer(
            base_channels * 2, base_channels * 4, time_embed_dim
        )
        self.down3 = nn.Conv2d(
            base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1
        )

        # Middle
        self.middle = self._make_layer(
            base_channels * 4, base_channels * 4, time_embed_dim
        )

        # Decoder
        self.up3 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 4, kernel_size=4, stride=2, padding=1
        )
        self.dec3 = self._make_layer(
            base_channels * 8, base_channels * 2, time_embed_dim
        )

        self.up2 = nn.ConvTranspose2d(
            base_channels * 2, base_channels * 2, kernel_size=4, stride=2, padding=1
        )
        self.dec2 = self._make_layer(base_channels * 4, base_channels, time_embed_dim)

        self.up1 = nn.ConvTranspose2d(
            base_channels, base_channels, kernel_size=4, stride=2, padding=1
        )
        self.dec1 = self._make_layer(base_channels * 2, base_channels, time_embed_dim)

        # output
        self.out = nn.Conv2d(base_channels, out_channels, 1)

    def _make_layer(self, in_channels, out_channels, embed_dim):
        return ResidualBlock(
            in_channels=in_channels, out_channels=out_channels, time_embed_dim=embed_dim
        )

    def forward(self, x, t):
        """
        Args:
            x: [B, C, H, W] noisy images
            t: [B] timesteps
        Returns:
            noise_pred: [B, C, H, W] predicted noise
        """
        t_emb = self.time_embed(t)

        x = self.conv_in(x)
        h1 = self.enc1(x, t_emb)
        h = self.down1(h1)

        h2 = self.enc2(h, t_emb)
        h = self.down2(h2)

        h3 = self.enc3(h, t_emb)
        h = self.down3(h3)

        h = self.middle(h, t_emb)

        h = self.up3(h)
        h = torch.cat([h, h3], dim=1)
        h = self.dec3(h, t_emb)

        h = self.up2(h)
        h = torch.cat([h, h2], dim=1)
        h = self.dec2(h, t_emb)

        h = self.up1(h)
        h = torch.cat([h, h1], dim=1)
        h = self.dec1(h, t_emb)

        return self.out(h)


def train_network(num_epochs=3, save_step=250):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleUNet(base_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    forward = ForwardDiffusion(timesteps=1000, schedule="cosine")
    loss_fn = SimplifiedDDPMLoss(forward, model)

    dataloader = CIFARDataLoader().get_dataloader(batch_size=128, shuffle=True)

    model.train()
    step = 0
    losses = []

    for epoch in tqdm(range(num_epochs)):
        for batch_idx, (x_0, label) in tqdm(enumerate(dataloader)):
            x_0 = x_0.to(device)

            # forward pass
            optimizer.zero_grad()
            loss, _, _, _ = loss_fn(model, x_0)

            # backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            losses.append(loss.item())
            step += 1

            if step % 100 == 0:
                avg_loss = np.mean(losses[-100:])
                print(
                    f"Epoch {epoch + 1}/{num_epochs} | Step {step} | Loss: {avg_loss:.4f}"
                )

            if step % save_step == 0:
                visualize_denoising(model, forward, device, step)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(losses, alpha=0.3)
    window = 100
    if len(losses) > window:
        moving_avg = np.convolve(losses, np.ones(window) / window, mode="valid")
        plt.plot(
            range(window - 1, len(losses)), moving_avg, linewidth=2, label="Moving Avg"
        )
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(losses, alpha=0.3)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss (Log Scale)")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "./outputs/epsilon_prediction_training_curve.png", dpi=150, bbox_inches="tight"
    )
    os.makedirs("./outputs/models", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "loss": losses[-1],
        },
        "./outputs/models/simple_unet.pt",
    )
    return model, losses


def visualize_denoising(model: SimpleUNet, forward: ForwardDiffusion, device, step):
    model.eval()
    dataset = CIFARDataLoader()

    x_0, label = dataset.get_samples(1)
    x_0 = x_0.to(device)

    timesteps = [0, 250, 500, 750, 999]

    fig, axes = plt.subplots(3, len(timesteps), figsize=(15, 9))

    with torch.no_grad():
        for idx, t_val in enumerate(timesteps):
            t = torch.tensor([t_val]).to(device)
            noise = torch.randn_like(x_0)
            x_t = forward.q_sample_direct(x_0, t, noise)

            noise_pred = model(x_t, t)

            def denorm(x):
                x = x * 0.5 + 0.5
                return torch.clamp(x, 0, 1)

            axes[0, idx].imshow(denorm(x_t[0]).cpu().permute(1, 2, 0).numpy())
            axes[0, idx].set_title(f"t={t_val}")
            axes[0, idx].axis("off")

            noise_vis = (noise[0] + 1) / 2
            axes[1, idx].imshow(noise_vis.cpu().permute(1, 2, 0).numpy())
            axes[1, idx].set_title("True Noise")
            axes[1, idx].axis("off")

            noise_pred_vis = (noise_pred[0] + 1) / 2
            axes[2, idx].imshow(noise_pred_vis.cpu().permute(1, 2, 0).numpy())
            axes[2, idx].set_title("Predicted Noise")
            axes[2, idx].axis("off")

    plt.suptitle(f"Noise Prediction at Step {step}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"./outputs/denoising_step_{step}.png", dpi=150, bbox_inches="tight")

    model.train()


if __name__ == "__main__":
    # train_network()
    train_network()
