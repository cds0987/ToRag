import os
import faiss


def _normalize_faiss_filename(filename: str) -> str:
    """
    Ensure filename has .faiss extension
    """
    if not filename.endswith(".faiss"):
        filename = filename + ".faiss"
    return filename


def save_faiss_local(index, directory: str, filename: str):
    """
    Save FAISS index to local filesystem.

    Parameters
    ----------
    index : faiss.Index
    directory : str
    filename : str
        name with or without .faiss extension
    """

    filename = _normalize_faiss_filename(filename)

    os.makedirs(directory, exist_ok=True)

    path = os.path.join(directory, filename)

    faiss.write_index(index, path)
    print(f"Saved index to {path}")

    return path


def load_faiss_local(directory: str, filename: str):
    """
    Load FAISS index from local filesystem.
    """

    filename = _normalize_faiss_filename(filename)

    path = os.path.join(directory, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Index not found: {path}")

    index = faiss.read_index(path)

    return index


import os
import tempfile
import faiss
from huggingface_hub import upload_file,hf_hub_download


def save_faiss_hf(index, repo_id: str, filename: str):
    """
    Save FAISS index to HuggingFace Hub dataset repo.
    """
    filename = _normalize_faiss_filename(filename)
    with tempfile.TemporaryDirectory() as tmpdir:

        path = os.path.join(tmpdir, filename)

        # save index locally first
        faiss.write_index(index, path)

        upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=path,
            path_in_repo=filename,
        )

    return f"{repo_id}/{filename}"



def load_faiss_hf(repo_id: str, filename: str):
    """
    Load FAISS index from HuggingFace Hub.
    """

    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename
    )

    index = faiss.read_index(path)

    return index