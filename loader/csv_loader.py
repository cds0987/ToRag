from .file_loader import FileLoader
from datasets import load_dataset


class CsvLoader(FileLoader):

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

        docs = []

        for i, row in enumerate(dataset):

            text = " ".join(
                f"{col}: {row[col]}" for col in dataset.column_names
            )

            docs.append({
                "text": text,
                "source": path,
                "row": i
            })

        return docs