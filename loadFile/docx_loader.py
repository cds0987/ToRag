from docx import Document
from .file_loader import FileLoader
class docxLoader(FileLoader):

    extensions = [".docx"]

    def _read(self, path):

        try:
            doc = Document(path)
        except Exception:
            return None

        text = "\n".join(p.text for p in doc.paragraphs)

        return {
            "doc": doc,
            "source": path
        }
        
from .load import loader_manager
# Register the loader
loader_manager.register_file_loader(docxLoader())