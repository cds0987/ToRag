import json
from .file_loader import FileLoader
class jsonLoader(FileLoader):

    extensions = [".json"]

    def _read(self, path):

        try:

            with open(path, "r", encoding="utf-8") as f:

                data = json.load(f)

        except Exception:

            return None

        docs = []

        if isinstance(data, list):

            for item in data:

                docs.append({
                    "text": json.dumps(item),
                    "source": path
                })

        else:

            docs.append({
                "text": json.dumps(data),
                "source": path
            })

        return docs
    
from .load import loader_manager
# Register the loader
loader_manager.register_file_loader(jsonLoader())