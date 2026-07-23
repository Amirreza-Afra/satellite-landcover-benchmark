from src.engine.validate import validate
from src.datasets.builder import build_dataset
from src.models.builder import build_model
from src.utils.device import get_device

from src.datasets.loader import build_dataloaders

from src.losses.builder import build_loss
from src.optimizers.builder import build_optimizer

from src.engine.train_one_epoch import train_one_epoch


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

        for epoch in range(self.cfg["train"]["epochs"]):

            train_loss = train_one_epoch(
                model=self.model,
                dataloader=self.train_loader,
                criterion=self.criterion,
                optimizer=self.optimizer,
                device=self.device,
                cfg=self.cfg,
            )

            
            val_loss = validate(
                model=self.model,
                dataloader=self.val_loader,
                criterion=self.criterion,
                device=self.device,
            )


            print(
                f"Epoch [{epoch+1}/{self.cfg['train']['epochs']}] "
                f"Train Loss: {train_loss:.4f}"
                f"Val Loss: {val_loss:.4f}"
            )
