import os
from typing import Callable
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image


class RockPaperScissorsDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        self.images: list[tuple[torch.Tensor, int]] = []
        for file in os.listdir(os.path.join(".", img_dir)):
            if file.endswith(".jpg"):
                image = decode_image(os.path.join(".", img_dir, file))

                if file.startswith("rock"):
                    label = 0  # rock=0
                elif file.startswith("paper"):
                    label = 1  # paper=1
                elif file.startswith("scissors"):
                    label = 2  # scissors=2
                else:
                    continue  # Skip files that don't match expected patterns

                self.images.append((image, label))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image, label = self.images[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
