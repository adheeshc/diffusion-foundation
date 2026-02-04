import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import CIFARDataLoader
from forward_process import ForwardDiffusion
from tqdm import tqdm


class SimplifiedDDPMLoss:
    """
    Implements Algorithm 1 from DDPM paper.

    This is the KEY training objective:
    L_simple(θ) = E_{t, x_0, ε} [||ε - ε_θ(x_t, t)||²]

    Algorithm 1 (Training):
    1. repeat
    2.   x_0 ~ q(x_0)                    [Sample from dataset]
    3.   t ~ Uniform({1, ..., T})        [Random timestep]
    4.   ε ~ N(0, I)                     [Sample noise]
    5.   Take gradient step on ∇_θ ||ε - ε_θ(√ᾱ_t·x_0 + √(1-ᾱ_t)·ε, t)||²
    6. until converged
    """

    def __init__(self, forward_diffusion, model):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.forward = forward_diffusion
        self.model = model.to(self.device)
        self.timesteps = forward_diffusion.timesteps
        self.loader = CIFARDataLoader()

    def __call__(self, model, x_0, reduction="mean"):
        batch_size = x_0.shape[0]

        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)

        noise = torch.randn_like(x_0)
        x_t = self.forward.q_sample_direct(x_0, t, noise)

        predicted_noise = model(x_t, t)

        if reduction == "none":
            loss = F.mse_loss(noise, predicted_noise, reduction="none")
        elif reduction == "sum":
            loss = F.mse_loss(noise, predicted_noise, reduction="sum")
        else:
            loss = F.mse_loss(noise, predicted_noise, reduction="mean")

        return loss, predicted_noise, noise, t

    def visualize_loss(self):

        x_0, _ = self.loader.get_samples(num_images=1)
        x_0 = x_0.to(self.device)
        timesteps_to_test = [0, 100, 250, 500, 750, 999]
        losses = []

        with torch.no_grad():
            for t_val in timesteps_to_test:
                t = torch.tensor([t_val], device=self.device)
                noise = torch.randn_like(x_0)
                x_t = self.forward.q_sample_direct(x_0, t, noise)
                pred = self.model(x_t, t)
                loss = F.mse_loss(noise, pred)
                losses.append(loss.item())

                print(f"t={t_val:4d}: loss={loss.item():.4f}")

        plt.figure(figsize=(10, 6))
        plt.plot(timesteps_to_test, losses, "o-", linewidth=2, markersize=8)
        plt.xlabel("Timestep t", fontsize=12)
        plt.ylabel("Loss (MSE)", fontsize=12)
        plt.title("Loss Variation Across Timesteps\n(Untrained Model)", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs("./outputs", exist_ok=True)
        plt.savefig("./outputs/loss_across_timesteps.png", dpi=150, bbox_inches="tight")


class SimpleNoisePredictor(nn.Module):
    """
    A simple CNN to predict noise.
    """

    def __init__(self, image_size=32, channels=3, time_embed_dim=128):
        super().__init__()

        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.conv1 = nn.Conv2d(channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, channels, 3, padding=1)

        self.act = nn.SiLU()

    def forward(self, x_t, t):
        """Predict noise from noisy image and timestep."""
        h = self.act(self.conv1(x_t))
        h = self.act(self.conv2(h))
        noise_pred = self.conv3(h)
        return noise_pred


def mini_training_loop(forward, model, loss_fn, optimizer, dataloader, num_steps=100):
    model.train()
    losses = []
    data_iter = iter(dataloader)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pbar = tqdm(range(num_steps), desc="Training")
    for step in pbar:
        try:
            x_0, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x_0, _ = next(data_iter)

        x_0 = x_0.to(device)

        optimizer.zero_grad()
        loss, _, _, _ = loss_fn(model, x_0)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(losses, alpha=0.6, label="Loss")
    window = 10
    if len(losses) > window:
        moving_avg = np.convolve(losses, np.ones(window) / window, mode="valid")
        plt.plot(
            range(window - 1, len(losses)),
            moving_avg,
            linewidth=2,
            label=f"Moving Avg (window={window})",
        )

    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training Loss (Mini Loop)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs("./outputs", exist_ok=True)
    plt.savefig("./outputs/mini_training_loss.png", dpi=150, bbox_inches="tight")

    print(f"\nInitial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Improvement: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")


if __name__ == "__main__":
    forward = ForwardDiffusion(timesteps=1000)
    model = SimpleNoisePredictor()
    loss_fn = SimplifiedDDPMLoss(forward, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataloader = CIFARDataLoader().get_dataloader()

    mini_training_loop(
        forward=forward,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        dataloader=dataloader,
    )
