from .base_loader import BaseLoader
import os
from concurrent.futures import ThreadPoolExecutor
import os


def read_file(path):

    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None
    
    
    
class txtload(BaseLoader):

    def _iter_paths(self, folder):

        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".txt"):
                    yield os.path.join(root, file)

    def load(self, folder, workers=8):

        with ThreadPoolExecutor(max_workers=workers) as executor:

            for text in executor.map(read_file, self._iter_paths(folder)):

                if text:
                    yield text

                    
                    
from .base_loader import BaseLoader
import os
import json
from concurrent.futures import ThreadPoolExecutor


class jsonLoader(BaseLoader):

    def _iter_paths(self, folder):

        for root, _, files in os.walk(folder):

            for file in files:

                if file.endswith(".json"):

                    yield os.path.join(root, file)

    def _read_json(self, path):

        try:

            with open(path, "r", encoding="utf-8") as f:

                data = json.load(f)

                if isinstance(data, list):

                    for item in data:
                        yield str(item)

                else:
                    yield json.dumps(data)

        except Exception:
            return

    def load(self, folder, workers=8):

        with ThreadPoolExecutor(max_workers=workers) as executor:

            for results in executor.map(self._read_json, self._iter_paths(folder)):

                if results:

                    for doc in results:
                        yield doc


class jsonlLoader(BaseLoader):

    def _iter_paths(self, folder):

        for root, _, files in os.walk(folder):

            for file in files:

                if file.endswith(".jsonl"):

                    yield os.path.join(root, file)

    def _read_file(self, path):

        with open(path, "r", encoding="utf-8") as f:

            for line in f:
                yield line.strip()

    def load(self, folder, workers=4):

        with ThreadPoolExecutor(max_workers=workers) as executor:

            for results in executor.map(self._read_file, self._iter_paths(folder)):

                for line in results:
                    yield line

from bs4 import BeautifulSoup
class htmlLoader(BaseLoader):

    def _read_html(self, path):

        html = read_file(path)

        if html:

            soup = BeautifulSoup(html, "html.parser")

            return soup.get_text()

        return None


    def _iter_paths(self, folder):

        for root, _, files in os.walk(folder):

            for file in files:

                if file.endswith(".html"):
                    yield os.path.join(root, file)


    def load(self, folder, workers=8):

        with ThreadPoolExecutor(max_workers=workers) as executor:

            for text in executor.map(self._read_html, self._iter_paths(folder)):

                if text:
                    yield text
