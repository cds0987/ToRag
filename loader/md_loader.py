
from importlib.abc import FileLoader

from loader.utils import read_file


class mdLoader(FileLoader):

    extensions = [".md"]

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
loader_manager.register_loader(mdLoader())