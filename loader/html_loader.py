from bs4 import BeautifulSoup
from .file_loader import FileLoader
from .utils import read_file

class HtmlLoader(FileLoader):

    extensions = [".html"]

    def _read(self, path):

        html = read_file(path)

        if not html:
            return None

        soup = BeautifulSoup(html, "html.parser")

        text = soup.get_text(separator=" ", strip=True)

        return {
            "text": text,
            "source": path
        }