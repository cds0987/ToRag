from .file_loader import FileLoader
from .utils import read_file


class txtLoader(FileLoader):

    extensions = [".txt"]

    def _read(self, path):

        text = read_file(path)

        if not text:
            return None

        return {
            "text": text,
            "source": path
        }

from .load import loader_manager
# Register the loader
loader_manager.register_file_loader(txtLoader())