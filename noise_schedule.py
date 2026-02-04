import os

import matplotlib.pyplot as plt
import numpy as np
import torch


class NoiseSchedule:
    """
    Compare different noise schedules for diffusion models + visualize
    - Linear schedule
    - Cosine schedule
    """

    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def linear_schedule(self, beta_start=0.0001, beta_end=0.02):
        """
        Linear beta schedule

        Formula: β_t = β_start + (β_end - β_start) * t / T

        Returns:
            betas: [timesteps] tensor of beta values
        """
        t = torch.linspace(0, self.timesteps - 1, self.timesteps, device=self.device)
        betas = beta_start + (beta_end - beta_start) * (t / (self.timesteps - 1))
        # betas = torch.linspace(beta_start, beta_end, self.timesteps)
        return betas

    def cosine_schedule(self, s=0.008):
        """
        Formula from Improved DDPM paper:
        α̅_t = f(t) / f(0)
        where f(t) = cos((t/T + s) / (1 + s) * π/2)²
        then β_t = 1 - (α̅_t / α̅_{t-1})

        Returns:
            betas: [timesteps] tensor of beta values

        Args:
            s (float, optional): Offset. Defaults to 0.008.
        """
        steps = self.timesteps + 1
        t = torch.linspace(0, self.timesteps, steps, device=self.device)
        func_t = torch.cos(((t / self.timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas = func_t / func_t[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        return torch.clamp(betas, min=0.0001, max=0.9999)

    def visualize_schedules(self):
        linear_betas = self.linear_schedule()
        cosine_betas = self.cosine_schedule()

        linear_alphas = 1 - linear_betas
        cosine_alphas = 1 - cosine_betas
        linear_alphas_cumprod = torch.cumprod(linear_alphas, dim=0)
        cosine_alphas_cumprod = torch.cumprod(cosine_alphas, dim=0)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # plot betas
        axes[0, 0].plot(linear_betas.cpu().numpy(), label="Linear")
        axes[0, 0].plot(cosine_betas.cpu().numpy(), label="Cosine")
        axes[0, 0].set_title("Beta Schedules")
        axes[0, 0].set_xlabel("Timestep")
        axes[0, 0].set_ylabel("Beta")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # plot alphas
        axes[0, 1].plot(linear_alphas_cumprod.cpu().numpy(), label="Linear")
        axes[0, 1].plot(cosine_alphas_cumprod.cpu().numpy(), label="Cosine")
        axes[0, 1].set_title("Cumulative Alpha")
        axes[0, 1].set_xlabel("Timestep")
        axes[0, 1].set_ylabel("Alpha")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # plot signal coefficient
        axes[1, 0].plot(torch.sqrt(linear_alphas_cumprod).cpu().numpy(), label="Linear")
        axes[1, 0].plot(torch.sqrt(cosine_alphas_cumprod).cpu().numpy(), label="Cosine")
        axes[1, 0].set_title("Signal Coefficient (sqrt(alpha))")
        axes[1, 0].set_xlabel("Timestep")
        axes[1, 0].set_ylabel("sqrt(alpha)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot Signal to Noise Ratio
        linear_snr = linear_alphas_cumprod / (1 - linear_alphas_cumprod)
        cosine_snr = cosine_alphas_cumprod / (1 - cosine_alphas_cumprod)
        axes[1, 1].plot(linear_snr.cpu().numpy(), label="Linear")
        axes[1, 1].plot(cosine_snr.cpu().numpy(), label="Cosine")
        axes[1, 1].set_title("Signal to Noise Ratio")
        axes[1, 1].set_xlabel("Timestep")
        axes[1, 1].set_ylabel("SNR")
        axes[1, 1].set_yscale("log")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs("./outputs", exist_ok=True)
        plt.savefig("./outputs/noise_schedules.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    schedule = NoiseSchedule(timesteps=1000)
    schedule.visualize_schedules()
