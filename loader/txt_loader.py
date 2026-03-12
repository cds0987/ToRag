from .file_loader import FileLoader
from .utils import read_file


class TxtLoader(FileLoader):

    extensions = [".txt"]

    def _read(self, path):

        text = read_file(path)

        if not text:
            return None

        return {
            "text": text,
            "source": path
        }
