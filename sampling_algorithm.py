import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import CIFARDataLoader
from ddpm_loss import SimplifiedDDPMLoss
from epsilon_prediction import SimpleUNet
from forward_process import ForwardDiffusion
from torchvision.utils import make_grid
from tqdm import tqdm

"""
1. Implement Sampling Algorithm from DDPM paper
2. Understand the reverse diffusion process step-by-step
3. Visualize progressive denoising
4. Generate samples from trained model
"""


class DDPMSampler:
    """
    Implements Algorithm 2 from DDPM paper: Sampling

    Algorithm 2:
    1. x_T ~ N(0, I)                          [Start from pure noise]
    2. for t = T, ..., 1 do
    3.   z ~ N(0, I) if t > 1, else z = 0     [Sample noise]
    4.   x_{t-1} = 1/√α_t * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t,t)) + σ_t * z
    5. end for
    6. return x_0
    """

    def __init__(self, model, forward_diffusion: ForwardDiffusion, device="cuda"):
        self.model = model
        self.forward = forward_diffusion
        self.device = device
        self.timesteps = forward_diffusion.timesteps

        self.betas = forward_diffusion.betas.to(device)
        self.alphas = forward_diffusion.alphas.to(device)
        self.alphas_cumprod = forward_diffusion.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = forward_diffusion.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = (
            forward_diffusion.sqrt_one_minus_alphas_cumprod.to(device)
        )
        self.posterior_variance = forward_diffusion.posterior_variance.to(device)

    @torch.no_grad()
    def p_sample_step(self, x_t, t):
        """
        Single denoising step: x_t → x_{t-1}

        x_{t-1} = 1/√α_t * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t,t)) + σ_t * z

        μ_θ = 1/√α_t * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t,t))
        x̂_0 = (x_t - √(1-ᾱ_t) * ε_θ) / √ᾱ_t

        Args:
            x_t: [B, C, H, W] noisy image at timestep t
            t: [B] current timestep

        Returns:
            x_t_minus_1: [B, C, H, W] slightly less noisy image
            pred_x0: [B, C, H, W] predicted clean image (for visualization)
        """
        noise_pred = self.model(x_t, t)

        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(
            -1, 1, 1, 1
        )

        # x̂_0 = (x_t - √(1-ᾱ_t) * ε_θ) / √ᾱ_t
        pred_x0 = (
            x_t - sqrt_one_minus_alphas_cumprod_t * noise_pred
        ) / sqrt_alphas_cumprod_t

        # μ_θ = 1/√α_t * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t,t))
        mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - (1 - alpha_t) / sqrt_one_minus_alphas_cumprod_t * noise_pred
        )

        # z ~ N(0, I) if t > 1, else z = 0     [Sample noise]
        if t[0] > 0:
            variance = self.posterior_variance[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x_t)

            x_t_minus_1 = mean + torch.sqrt(variance) * noise
        else:
            x_t_minus_1 = mean

        return x_t_minus_1, pred_x0

    @torch.no_grad()
    def sample(self, shape, return_intermediates=False):
        """
        Full sampling procedure (Algorithm 2)

        1. x_T ~ N(0, I)                          [Start from pure noise]
        2. for t = T, ..., 1 do
        3.   z ~ N(0, I) if t > 1, else z = 0     [Sample noise]
        4.   x_{t-1} = 1/√α_t * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t,t)) + σ_t * z
        5. end for
        6. return x_0

        Args:
            shape: (batch_size, channels, height, width)
            return_intermediates: If True, return images at intermediate steps

        Returns:
            samples: [B, C, H, W] generated images
            intermediates: List of intermediate images (if return_intermediates=True)
        """
        self.model.eval()

        # line 1.
        x_t = torch.randn(shape, device=self.device)

        intermediates = []
        if return_intermediates:
            save_steps = [999, 750, 500, 250, 100, 50, 0]

        # line 2.
        for t_idx in tqdm(
            reversed(range(self.timesteps)), desc="Sampling", total=self.timesteps
        ):
            t = torch.full((shape[0],), t_idx, device=self.device, dtype=torch.long)
            # line 3+4
            x_t, pred_x0 = self.p_sample_step(x_t, t)

            if return_intermediates and t_idx in save_steps:
                intermediates.append(
                    {
                        "t": t_idx,
                        "x_t": x_t.cpu().clone(),
                        "pred_x0": pred_x0.cpu().clone(),
                    }
                )

        # line 6
        if return_intermediates:
            return x_t, intermediates
        return x_t

    def visualize_sampling(self, samples, intermediates):

        num_samples = len(samples)
        num_steps = len(intermediates)
        fig, axes = plt.subplots(
            num_samples, num_steps, figsize=(3 * num_steps, 3 * num_samples)
        )

        def denorm(x):
            """Denormalize from [-1, 1] to [0, 1]"""
            return (x * 0.5 + 0.5).clamp(0, 1)

        for sample_idx in range(num_samples):
            for step_idx, intermediate in enumerate(intermediates):
                ax = axes[sample_idx, step_idx] if num_samples > 1 else axes[step_idx]

                # Show x_t (current noisy image)
                img = denorm(intermediate["x_t"][sample_idx])  # type: ignore
                img = img.permute(1, 2, 0).numpy()

                ax.imshow(img)
                if sample_idx == 0:
                    ax.set_title(f"t={intermediate['t']}", fontsize=10)  # type: ignore
                ax.axis("off")

        plt.suptitle("Progressive Denoising (Algorithm 2)", fontsize=14)
        plt.tight_layout()
        plt.savefig("./outputs/sampling_process.png", dpi=150, bbox_inches="tight")

    def visualize_predicted_x0(self, samples, intermediates):

        num_steps = len(intermediates)
        num_samples = len(samples)
        fig, axes = plt.subplots(
            num_samples, num_steps, figsize=(3 * num_steps, 3 * num_samples)
        )

        def denorm(x):
            return (x * 0.5 + 0.5).clamp(0, 1)

        for sample_idx in range(num_samples):
            for step_idx, intermediate in enumerate(intermediates):
                ax = axes[sample_idx, step_idx] if num_samples > 1 else axes[step_idx]

                # Show predicted x_0
                img = denorm(intermediate["pred_x0"][sample_idx])  # type: ignore
                img = img.permute(1, 2, 0).numpy()

                ax.imshow(img)
                if sample_idx == 0:
                    ax.set_title(f"t={intermediate['t']}\n(pred x₀)", fontsize=9)  # type: ignore
                ax.axis("off")

        plt.suptitle("Predicted Clean Image (x₀) During Sampling", fontsize=14)
        plt.tight_layout()
        plt.savefig("./outputs/predicted_x0.png", dpi=150, bbox_inches="tight")

    def generate_sample_grid(self, samples, intermediates):
        """Generate a grid of samples."""

        # Denormalize
        samples = (samples * 0.5 + 0.5).clamp(0, 1)  # type: ignore

        # Create grid
        grid = make_grid(samples.cpu(), nrow=8, padding=2)

        # Plot
        plt.figure(figsize=(15, 15))
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.axis("off")
        plt.title("Generated Samples", fontsize=16)
        plt.tight_layout()
        plt.savefig("./outputs/sample_grid.png", dpi=150, bbox_inches="tight")


class ImprovedDDPMSampler(DDPMSampler):
    """
    DDPM sampler with different variance options.

    From "Improved Denoising Diffusion Probabilistic Models:
    - Can use β_t (more stochastic) or β̃_t (less stochastic)
    - β̃_t = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) * β_t
    """

    def __init__(
        self, model, forward_diffusion, variance_type="posterior", device="cuda"
    ):
        """
        Args:
            variance_type: 'posterior' (β̃_t) or 'beta' (β_t)
        """
        super().__init__(model, forward_diffusion, device)
        self.variance_type = variance_type

    @torch.no_grad()
    def p_sample_step(self, x_t, t):
        """
        Single denoising step: x_t → x_{t-1}

        x_{t-1} = 1/√α_t * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t,t)) + σ_t * z

        μ_θ = 1/√α_t * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t,t))
        x̂_0 = (x_t - √(1-ᾱ_t) * ε_θ) / √ᾱ_t

        Args:
            x_t: [B, C, H, W] noisy image at timestep t
            t: [B] current timestep

        Returns:
            x_t_minus_1: [B, C, H, W] slightly less noisy image
            pred_x0: [B, C, H, W] predicted clean image (for visualization)
        """
        noise_pred = self.model(x_t, t)

        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(
            -1, 1, 1, 1
        )

        # x̂_0 = (x_t - √(1-ᾱ_t) * ε_θ) / √ᾱ_t
        pred_x0 = (
            x_t - sqrt_one_minus_alphas_cumprod_t * noise_pred
        ) / sqrt_alphas_cumprod_t

        # μ_θ = 1/√α_t * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t,t))
        mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - (1 - alpha_t) / sqrt_one_minus_alphas_cumprod_t * noise_pred
        )

        # z ~ N(0, I) if t > 1, else z = 0     [Sample noise]
        if self.variance_type == "posterior":
            variance = self.posterior_variance[t].view(-1, 1, 1, 1)
        else:
            variance = beta_t

        if t[0] > 0:
            noise = torch.randn_like(x_t)
            x_t_minus_1 = mean + torch.sqrt(variance) * noise
        else:
            x_t_minus_1 = mean

        return x_t_minus_1, pred_x0


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleUNet().to(device)
    model_path = "./outputs/models/simple_unet.pt"
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded from step {checkpoint['step']}")

    forward = ForwardDiffusion()
    sampler = DDPMSampler(model=model, forward_diffusion=forward, device=device)

    samples, intermediates = sampler.sample((4, 3, 32, 32), return_intermediates=True)
    sampler.visualize_sampling(samples, intermediates)
    sampler.visualize_predicted_x0(samples, intermediates)
    sampler.generate_sample_grid(samples, intermediates)
