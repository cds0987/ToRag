import pandas as pd
from datasets import Dataset
from .file_loader import FileLoader


class excelLoader(FileLoader):

    extensions = [".xlsx", ".xls"]

    def _read(self, path):

        try:
            df = pd.read_excel(path)
        except Exception:
            return None

        return Dataset.from_pandas(df)

from .load import loader_manager
# Register the loader
loader_manager.register_loader(excelLoader())