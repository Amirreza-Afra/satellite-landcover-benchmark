from src.configs.loader import load_config


def main():
    cfg = load_config("src/configs/base.yaml")

    print(cfg)


if __name__ == "__main__":
    main()