from src.configs.loader import load_config
from src.datasets.builder import build_dataset

# مهم: این import برای ثبت شدن دیتاست در Registry لازم است
import src.datasets.eurosat


def main():

    cfg = load_config("src/configs/base.yaml")

    dataset = build_dataset(cfg)

    print(dataset)


if __name__ == "__main__":
    main()