"""
Conditional MNIST -> MNIST bridge dataset.

Each training example is a triple (x_0, x_1, y) where:
  - y ~ Uniform{0..9}                       target digit label
  - x_0 ~ p(image | digit = y)              target-digit image (what reverse sampler generates)
  - x_1 ~ p(image)                          random-digit image (reverse-sampler start)

Reverse sampler at inference: given (x_1, y), generate an image of digit y
that begins its trajectory at x_1.
"""

import os
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


class MNIST2MNISTDataset(Dataset):
    """Conditional MNIST bridge: x_0 = target-digit image, x_1 = random image, y = target digit."""

    def __init__(
        self,
        data_dir: str = "/orcd/data/omarabu/001/njwfish/counting_flows/datasets/data/mnist",
        seed: int = 42,
        train: bool = True,
    ):
        self.data_dir = data_dir
        self.in_chans = 1
        self.img_size = 28
        self.data_dim = 28 * 28  # for main.py's logging/shape checks

        processed_path = os.path.join(
            data_dir, f"processed_mnist_{'train' if train else 'test'}.pt"
        )
        if os.path.exists(processed_path):
            cached = torch.load(processed_path)
            self.data = cached["data"]
            self.labels = cached["labels"]
        else:
            os.makedirs(data_dir, exist_ok=True)
            mnist = MNIST(data_dir, train=train, download=True)
            self.data = mnist.data.float() / 255.0 * 2.0 - 1.0  # -> [-1, 1]
            self.data = self.data.unsqueeze(1)                   # (N, 1, 28, 28)
            self.labels = mnist.targets.long()
            torch.save({"data": self.data, "labels": self.labels}, processed_path)

        # Bucket indices by digit for fast per-class sampling.
        self.indices_per_class = [
            (self.labels == c).nonzero(as_tuple=True)[0] for c in range(10)
        ]

        # Deterministic pre-sample of target labels and x_1 indices (so __getitem__
        # is cheap and reproducible across epochs within a seed).
        g = torch.Generator().manual_seed(seed)
        self.num_classes = 10
        self.size = len(self.data)
        self.target_labels = torch.randint(0, self.num_classes, (self.size,), generator=g)
        # For each sample i, pick a random index of (target_labels[i])-class image.
        self.x0_indices = torch.empty(self.size, dtype=torch.long)
        for c in range(self.num_classes):
            mask = (self.target_labels == c)
            n = int(mask.sum().item())
            if n == 0:
                continue
            pool = self.indices_per_class[c]
            pick = pool[torch.randint(0, len(pool), (n,), generator=g)]
            self.x0_indices[mask] = pick

        # x_1: for each sample i, use the dataset's own i-th image as the source (generic MNIST).
        # Equivalent to x_1 ~ MNIST marginal, iid of y.
        self.x1_indices = torch.arange(self.size)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x_0": self.data[self.x0_indices[idx]],
            "x_1": self.data[self.x1_indices[idx]],
            "y": self.target_labels[idx],
        }
