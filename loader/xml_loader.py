import xml.etree.ElementTree as ET

from .file_loader import FileLoader
class xmlLoader(FileLoader):

    extensions = [".xml"]

    def _read(self, path):

        try:
            tree = ET.parse(path)
            root = tree.getroot()
        except Exception:
            return None

        text = " ".join(elem.text for elem in root.iter() if elem.text)

        return {
            "text": text,
            "source": path
        }

from .load import loader_manager
# Register the loader
loader_manager.register_loader(xmlLoader())