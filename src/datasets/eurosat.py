from torch.utils.data import random_split
from torchvision.datasets import EuroSAT

from src.transforms.builder import build_transforms

from .registry import register_dataset

@register_dataset("eurosat")
class EuroSATDataset:

    def __init__(self, cfg):

        self.cfg = cfg

        train_transform, val_transform = build_transforms(cfg)

        dataset = EuroSAT(
            root=cfg["data"]["root"],
            download=True,
            transform=train_transform
        )

        train_size = int(cfg["data"]["train_split"] * len(dataset))
        val_size = len(dataset) - train_size

        self.train_dataset, self.val_dataset = random_split(
            dataset,
            [train_size, val_size]
        )

        print(f"Train Dataset: {len(self.train_dataset)}")
        print(f"Validation Dataset: {len(self.val_dataset)}")