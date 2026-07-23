from pathlib import Path
import torch


class CheckpointManager:

    def __init__(self, cfg):
        checkpoint_dir = cfg["paths"]["checkpoint_dir"]

        if checkpoint_dir == "":
            checkpoint_dir = "checkpoints"

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model,
        optimizer,
        epoch,
        val_loss,
        filename="last.pth",
    ):

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        }

        torch.save(
            checkpoint,
            self.checkpoint_dir / filename,
        )