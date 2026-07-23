from .registry import METRIC_REGISTRY


def build_metrics(cfg):
    metrics = {}
    for name in cfg["metrics"]:
        metric_class = METRIC_REGISTRY[name]
        metrics[name] = metric_class()
    return metrics