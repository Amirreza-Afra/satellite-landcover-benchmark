from .train import get_train_transforms
from .val import get_val_transforms


def build_transforms(cfg):

    image_size = cfg["data"]["image_size"]

    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)

    return train_transform, val_transform