from torch.utils.data import DataLoader


def build_dataloaders(dataset, cfg):

    train_loader = DataLoader(
        dataset.train_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
    )

    val_loader = DataLoader(
        dataset.val_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )

    return train_loader, val_loader