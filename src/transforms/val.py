import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_val_transforms(image_size: int):

    return A.Compose([
        A.Resize(image_size, image_size),

        A.Normalize(),

        ToTensorV2()
    ])