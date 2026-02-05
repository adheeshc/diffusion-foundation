import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CIFARDataLoader:
    CLASS_NAMES = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    def __init__(self, root="../../../Datasets", train=True, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=False,
            transform=self.transform,
        )

    def get_samples(self, num_images, step=1000):
        """Get sample images and labels from the dataset."""
        images = []
        labels = []
        for i in range(num_images):
            img, label = self.dataset[i * step]
            images.append(img)
            labels.append(self.CLASS_NAMES[label])
        images = torch.stack(images).to(self.device)
        return images, labels

    def get_dataloader(
        self, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    ):
        """Get a PyTorch DataLoader for batched iteration."""
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
