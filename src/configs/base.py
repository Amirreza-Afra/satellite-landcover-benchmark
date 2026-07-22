from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 42

    image_size: int = 224

    batch_size: int = 32

    epochs: int = 10

    learning_rate: float = 1e-3

    num_workers: int = 2

    num_classes: int = 10

    pretrained: bool = True

    model_name: str = "resnet18"

    dataset_root: str = "datasets/EuroSAT"