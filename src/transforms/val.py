from torchvision import transforms


def get_val_transforms(image_size):

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])