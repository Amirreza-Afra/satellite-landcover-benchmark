import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_train_transforms(image_size: int):

    return A.Compose([
        A.Resize(image_size, image_size),

        A.HorizontalFlip(p=0.5),

        A.Normalize(),

        ToTensorV2()
    ])