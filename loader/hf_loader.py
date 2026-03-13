from datasets import load_dataset
from .base_loader import BaseLoader


class HuggingFaceLoader(BaseLoader):

    def load(self, dataset_name, split="train", **kwargs):

        try:
            dataset = load_dataset(dataset_name, split=split, **kwargs)
        except Exception:
            return None

        yield dataset
from .load import loader_manager
# Register the loader
loader_manager.register_loader(HuggingFaceLoader())