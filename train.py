from src.configs.loader import load_config

from src.engine.trainer import Trainer

import src.datasets
import src.models


def main():

    cfg = load_config("src/configs/base.yaml")

    trainer = Trainer(cfg)

    trainer.fit()


if __name__ == "__main__":
    main()