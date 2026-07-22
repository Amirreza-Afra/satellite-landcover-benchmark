from src.configs.loader import load_config

from src.datasets.builder import build_dataset
from src.models.builder import build_model

import src.datasets
import src.models


def main():

    cfg = load_config("src/configs/base.yaml")

    dataset = build_dataset(cfg)

    model = build_model(cfg)

    print(dataset)
    print(model)


if __name__ == "__main__":
    main()