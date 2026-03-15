from datasets import load_dataset
from .base_loader import BaseLoader


class hfdatasetLoader(BaseLoader):

    def load(self, dataset_name, split="train", **kwargs):

        try:
            dataset = load_dataset(dataset_name, split=split, **kwargs)
        except Exception:
            return None

        yield dataset
from .load import loader_manager
# Register the loader
loader_manager.register_hf_loader(hfdatasetLoader())

