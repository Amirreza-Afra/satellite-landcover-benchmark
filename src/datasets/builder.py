from .registry import DATASET_REGISTRY


def build_dataset(cfg):
    dataset_name = cfg["dataset"]["name"]

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{dataset_name}' is not registered.")

    dataset_class = DATASET_REGISTRY[dataset_name]

    return dataset_class(cfg)