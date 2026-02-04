import os

import matplotlib.pyplot as plt
import torch
from dataloader import CIFARDataLoader
from noise_schedule import NoiseSchedule


class ForwardDiffusion:
    """
    Complete forward diffusion implementation

    Pipeline:
    1. Implement q(x_t | x_0) - direct sampling
    2. Implement q(x_t | x_{t-1}) - iterative sampling
    3. Verify they produce the same distribution
    4. Implement posterior q(x_{t-1} | x_t, x_0)
    """

    def __init__(self, timesteps=1000, schedule="linear"):
        self.timesteps = timesteps
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loader = CIFARDataLoader()

        noise_schedule = NoiseSchedule(timesteps)
        self.schedule = schedule
        if self.schedule == "linear":
            betas = noise_schedule.linear_schedule()
        else:
            betas = noise_schedule.cosine_schedule()

        self.betas = betas.to(self.device)
        self.alphas = (1 - betas).to(self.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod).to(
            self.device
        )

    def q_sample_direct(self, x_0, t, noise=None):
        """Sample from q(x_t | x_0) directly

        Args:
            x_0: Clean image [B, C, H, W]
            t: Timestep

        Returns:
            Noisy image x_t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def q_sample_iterative(self, x_0, t):
        """
        Sample from q(x_t | x_0) by iterating through all steps. This is slow but helps understand the Markov process.
        Formula: x_t = √(1 - β_t) * x_{t-1} + √β_t * ε_t

        Args:
            x_0: Clean image [B, C, H, W]
            t: Timestep

        Returns:
            Noisy image x_t
        """
        x_t = x_0.clone()
        for i in range(t.item() + 1):
            noise = torch.randn_like(x_0).to(self.device)
            beta_i = self.betas[i].view(-1, 1, 1, 1)
            x_t = torch.sqrt(1 - beta_i) * x_t + torch.sqrt(beta_i) * noise
        return x_t

    def q_posterior(self, x_0, x_t, t):
        """Compute posterior q(x_{t-1} | x_t, x_0).

        Formula:
          q(x_{t-1} | x_t, x_0) = N(x_{t-1}; μ̃_t, β̃_t)

          μ̃_t = (√α̅_{t-1} * β_t) / (1 - α̅_t) * x_0
               + (√α_t * (1 - α̅_{t-1})) / (1 - α̅_t) * x_t

          β̃_t = (1 - α̅_{t-1}) / (1 - α̅_t) * β_t

        Where:
          α_t = 1 - β_t
          α̅_t = ∏_{s=1}^{t} α_s  (cumulative product)

        Returns mean and variance.
        """
        alpha_t = self.alphas[t].view(-1, 1, 1, 1).to(self.device)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_cumprod_t_prev = (
            self.alphas_cumprod[t - 1].view(-1, 1, 1, 1)
            if t > 0
            else torch.tensor(1.0).to(self.device)
        )
        sqrt_alpha_cumprod_t_prev = torch.sqrt(alpha_cumprod_t_prev)
        beta_t = self.betas[t].view(-1, 1, 1, 1).to(self.device)

        posterior_mean = (sqrt_alpha_cumprod_t_prev * beta_t * x_0) / (
            1 - alpha_cumprod_t
        ) + (torch.sqrt(alpha_t) * (1 - alpha_cumprod_t_prev) * x_t) / (
            1 - alpha_cumprod_t
        )

        posterior_variance = (
            (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t)
        ) * beta_t

        return posterior_mean, posterior_variance

    def visualize_diffusion_steps(self, num_images=4, num_steps=8):
        images, labels = self.loader.get_samples(num_images=num_images)
        timesteps = torch.linspace(0, self.timesteps - 1, num_steps).long()

        fig, axes = plt.subplots(
            num_images, num_steps, figsize=(num_steps * 2, num_images * 2)
        )

        for i in range(num_images):
            for j, t in enumerate(timesteps):
                t_batch = t.unsqueeze(0)
                noisy = self.q_sample_direct(images[i : i + 1], t_batch)

                img = noisy[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()

                axes[i, j].imshow(img)
                axes[i, j].axis("off")
                if i == 0:
                    axes[i, j].set_title(f"t={t.item()}", fontsize=10)

        plt.suptitle("Forward Diffusion Process", fontsize=14)
        plt.tight_layout()
        os.makedirs("./outputs", exist_ok=True)
        plt.savefig(
            f"./outputs/forward_process_{self.schedule}.png",
            dpi=150,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    forward = ForwardDiffusion(timesteps=1000, schedule="linear")
    forward.visualize_diffusion_steps(num_images=4, num_steps=8)
