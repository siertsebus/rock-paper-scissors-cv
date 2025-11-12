import os
from typing import Callable
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torch.utils.data import Subset


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
                image = decode_image(os.path.join(".", img_dir, file)).float() / 255.0

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

    def balanced_split(
        self, ratios: list[float]
    ) -> list[Subset[tuple[torch.Tensor, int]]]:
        """Splits the dataset into balanced subsets according to the given ratios."""
        # Group indices by class
        class_indices: dict[int, list[int]] = {0: [], 1: [], 2: []}
        for idx, (_, label) in enumerate(self.images):
            class_indices[label].append(idx)

        # Calculate number of samples per class for each split
        assert sum(ratios) - 1.0 < 1e-6, "Ratios must sum to 1"

        splits: list[list[int]] = [[] for _ in ratios]
        for label, indices in class_indices.items():
            total_count = len(indices)
            start = 0
            for split_idx, ratio in enumerate(ratios):
                count = int(total_count * ratio)
                splits[split_idx].extend(indices[start : start + count])
                start += count

        # Create Subset datasets
        return [Subset(self, split) for split in splits]
