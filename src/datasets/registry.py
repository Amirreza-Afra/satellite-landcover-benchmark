DATASET_REGISTRY = {}


def register_dataset(name):
    def decorator(dataset_class):
        DATASET_REGISTRY[name] = dataset_class
        return dataset_class

    return decorator