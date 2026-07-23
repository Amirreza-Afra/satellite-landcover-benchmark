from src.datasets.builder import build_dataset
from src.models.builder import build_model
from src.utils.device import get_device

from src.datasets.loader import build_dataloaders

from src.losses.builder import build_loss
from src.optimizers.builder import build_optimizer


class Trainer:

    def __init__(self, cfg):
        self.cfg = cfg
        # Device
        self.device = get_device()
        # Components
        self.dataset = build_dataset(cfg)
        self.model = build_model(cfg)
        # Move model to device
        self.model.to(self.device)
        self.train_loader, self.val_loader = build_dataloaders(
            self.dataset,
            cfg
        )
        self.criterion = build_loss(cfg)
        self.optimizer = build_optimizer(cfg, self.model.parameters())
        print("Trainer Initialized")

    def fit(self):
        print("Training Started...")
        print(f"Device: {self.device}")
        print(self.criterion)
        print(self.optimizer)
        images, labels = next(iter(self.train_loader))
        print(images.shape)
        print(labels.shape)


