from .file_loader import FileLoader


class jsonlLoader(FileLoader):

    extensions = [".jsonl"]

    def _read(self, path):

        docs = []

        try:

            with open(path, "r", encoding="utf-8") as f:

                for line in f:

                    docs.append({
                        "text": line.strip(),
                        "source": path
                    })

        except Exception:

            return None

        return docs

from .load import loader_manager
# Register the loader
loader_manager.register_loader(jsonlLoader())