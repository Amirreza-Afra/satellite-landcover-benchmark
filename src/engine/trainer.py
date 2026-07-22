from src.datasets.builder import build_dataset
from src.models.builder import build_model
from src.utils.device import get_device


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
        print("Trainer Initialized")

    def fit(self):
        print("Training Started...")
        print(f"Device: {self.device}")
        print(self.dataset)
        print(self.model)