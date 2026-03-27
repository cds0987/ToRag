CLASS_REGISTRY = {}

import inspect

def filter_kwargs(cls, metadata: dict):
    sig = inspect.signature(cls.__init__)
    valid_params = sig.parameters.keys()

    return {
        k: v for k, v in metadata.items()
        if k in valid_params
    }

def from_metadata(metadata: dict):
    cls_name = metadata.get("class")

    if cls_name not in CLASS_REGISTRY:
        raise ValueError(f"Unknown class: {cls_name}")

    cls = CLASS_REGISTRY[cls_name]

    kwargs = filter_kwargs(cls, metadata)

    obj = cls(**kwargs)

    return obj