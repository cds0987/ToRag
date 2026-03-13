from .file_loader import FileLoader
from datasets import load_dataset


class csvLoader(FileLoader):

    extensions = [".csv"]

    def _read(self, path):

        try:

            dataset = load_dataset(
                "csv",
                data_files=path,
                split="train"
            )

        except Exception:
            return None

        return dataset
    
from .load import loader_manager
# Register the loader
loader_manager.register_file_loader(csvLoader())