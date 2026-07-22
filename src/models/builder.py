from .registry import MODEL_REGISTRY


def build_model(cfg):

    model_name = cfg["model"]["name"]

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' is not registered."
        )

    model_class = MODEL_REGISTRY[model_name]

    return model_class(cfg)