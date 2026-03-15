from datasets import load_dataset
from .file_loader import FileLoader


class parquetLoader(FileLoader):

    extensions = [".parquet"]

    def _read(self, path):

        try:
            dataset = load_dataset(
                "parquet",
                data_files=path,
                split="train"
            )
        except Exception:
            return None

        return dataset

from .load import loader_manager
# Register the loader
loader_manager.register_file_loader(parquetLoader())