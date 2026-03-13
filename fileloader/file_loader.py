from .base_loader import BaseLoader
import os
from concurrent.futures import ThreadPoolExecutor

class FileLoader(BaseLoader):

    extensions = []

    def _iter_paths(self, folder):

        for root, _, files in os.walk(folder):

            for file in files:

                if any(file.endswith(ext) for ext in self.extensions):

                    yield os.path.join(root, file)

    def _read(self, path):
        raise NotImplementedError

    def load(self, folder, workers=8):

        with ThreadPoolExecutor(max_workers=workers) as executor:

            for result in executor.map(self._read, self._iter_paths(folder)):

                if not result:
                    continue

                if isinstance(result, list):

                    for doc in result:
                        yield doc

                else:
                    yield result