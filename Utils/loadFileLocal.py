import json
import os
from typing import Any

def load_json_local(directory: str,filename: str,) -> Any:
    """
    Load a JSON file from local disk.
    """
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


import os
import json
import pandas as pd


class FileLoader:
    _cache = {}

    # -------------------------
    # Path resolver
    # -------------------------
    @staticmethod
    def _resolve_path(path=None, directory=None, filename=None):
        if path:
            return os.path.abspath(path)

        if directory and filename:
            return os.path.abspath(os.path.join(directory, filename))

        if filename:
            return os.path.abspath(filename)

        raise ValueError("Must provide either `path` or (`directory` + `filename`)")

    # -------------------------
    # Public API
    # -------------------------
    @staticmethod
    def load(path=None, directory=None, filename=None, use_cache=True, **kwargs):
        full_path = FileLoader._resolve_path(path, directory, filename)

        ext = os.path.splitext(full_path)[-1].lower()

        if use_cache and full_path in FileLoader._cache:
            return FileLoader._cache[full_path]

        loader_map = {
            ".csv": FileLoader.load_csv,
            ".tsv": FileLoader.load_tsv,
            ".xlsx": FileLoader.load_excel,
            ".xls": FileLoader.load_excel,
            ".json": FileLoader.load_json,
            ".jsonl": FileLoader.load_jsonl,
            ".txt": FileLoader.load_text,
            ".md": FileLoader.load_text,
            ".html": FileLoader.load_text,
            ".xml": FileLoader.load_text,
            ".parquet": FileLoader.load_parquet,
            ".docx": FileLoader.load_docx,
            ".pdf": FileLoader.load_pdf,
        }

        if ext not in loader_map:
            raise ValueError(f"Unsupported file type: {ext}")

        data = loader_map[ext](full_path, **kwargs)

        if use_cache:
            FileLoader._cache[full_path] = data

        return data

    # -------------------------
    # Individual loaders
    # -------------------------
    @staticmethod
    def load_csv(path, **kwargs):
        return pd.read_csv(path, **kwargs)

    @staticmethod
    def load_tsv(path, **kwargs):
        return pd.read_csv(path, sep="\t", **kwargs)

    @staticmethod
    def load_excel(path, **kwargs):
        return pd.read_excel(path, **kwargs)

    @staticmethod
    def load_json(path, **kwargs):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def load_jsonl(path, **kwargs):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    @staticmethod
    def load_text(path, **kwargs):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def load_parquet(path, **kwargs):
        return pd.read_parquet(path, **kwargs)

    @staticmethod
    def load_docx(path, **kwargs):
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)

    @staticmethod
    def load_pdf(path, **kwargs):
        from PyPDF2 import PdfReader
        reader = PdfReader(path)
        return "\n".join(p.extract_text() or "" for p in reader.pages)

    # -------------------------
    # Optional utilities
    # -------------------------
    @staticmethod
    def clear_cache():
        FileLoader._cache.clear()

    @staticmethod
    def exists(path=None, directory=None, filename=None):
        full_path = FileLoader._resolve_path(path, directory, filename)
        return os.path.exists(full_path)