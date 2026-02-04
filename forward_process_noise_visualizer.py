import os

import matplotlib.pyplot as plt
import torch
from dataloader import CIFARDataLoader
from noise_schedule import NoiseSchedule


class ForwardProcessNoiseVisualizer:
    """
    Visualize how images become noise over time.

    Pipeline:
    1. Load sample images
    2. Apply forward process at different timesteps
    3. Create visualization showing progressive noise
    4. Understand the corruption process
    """

    def __init__(self, timesteps=1000, schedule="linear"):
        self.timesteps = timesteps
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        noise_schedule = NoiseSchedule(timesteps)
        self.schedule = schedule
        if self.schedule == "linear":
            betas = noise_schedule.linear_schedule()
        else:
            betas = noise_schedule.cosine_schedule()

        alphas = 1 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod).to(
            self.device
        )
        self.loader = CIFARDataLoader()

    def add_noise(self, x_0, t):
        """
        Apply forward diffusion.

        Formula: x_t = √α̅_t * x_0 + √(1 - α̅_t) * ε

        Args:
            x_0: Clean image [B, C, H, W]
            t: Timestep

        Returns:
            Noisy image x_t
        """
        noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def visualize_diffusion_steps(self, num_images=4, num_steps=8):
        images, labels = self.loader.get_samples(num_images=num_images)
        timesteps = torch.linspace(0, self.timesteps - 1, num_steps).long()

        fig, axes = plt.subplots(
            num_images, num_steps, figsize=(num_steps * 2, num_images * 2)
        )

        for i in range(num_images):
            for j, t in enumerate(timesteps):
                t_batch = t.unsqueeze(0)
                noisy = self.add_noise(images[i : i + 1], t_batch)

                img = noisy[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()

                axes[i, j].imshow(img)
                axes[i, j].axis("off")
                if i == 0:
                    axes[i, j].set_title(f"t={t.item()}", fontsize=10)

        plt.suptitle("Forward Diffusion Process", fontsize=14)
        plt.tight_layout()
        os.makedirs("./outputs", exist_ok=True)
        plt.savefig(
            f"./outputs/forward_process_{self.schedule}_1.png",
            dpi=150,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    visualizer = ForwardProcessNoiseVisualizer(timesteps=1000, schedule="linear")
    visualizer.visualize_diffusion_steps(num_images=4, num_steps=8)
