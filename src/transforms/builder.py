from .train import build_train_transforms
from .val import build_val_transforms


def build_transforms(cfg):

    image_size = cfg["train"]["image_size"]

    train_transform = build_train_transforms(image_size)

    val_transform = build_val_transforms(image_size)

    return train_transform, val_transform