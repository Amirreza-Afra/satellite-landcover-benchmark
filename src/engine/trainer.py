from src.datasets.builder import build_dataset
from src.models.builder import build_model


class Trainer:

    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset = build_dataset(cfg)
        self.model = build_model(cfg)
        print("Trainer Initialized")


    def fit(self):
        print("Training Started...")
        print(self.dataset)
        print(self.model)