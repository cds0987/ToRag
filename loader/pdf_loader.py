import pdfplumber
from .file_loader import FileLoader
class pdfLoader(FileLoader):

    extensions = [".pdf"]

    def _read(self, path):

        docs = []

        try:
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()

                    if text:
                        docs.append({
                            "text": text,
                            "source": path,
                            "page": i
                        })
        except Exception:
            return None

        return docs

from .load import loader_manager
# Register the loader
loader_manager.register_loader(pdfLoader())