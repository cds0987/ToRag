import os
import faiss


def _normalize_faiss_filename(filename: str) -> str:
    """
    Ensure filename has .faiss extension
    """
    if not filename.endswith(".faiss"):
        filename = filename + ".faiss"
    return filename



def _prepare_index_for_saving(index, clone: bool = False):
    """
    Ensure index is CPU and safe to save.

    Handles:
    - GpuIndex
    - IndexPreTransform (wrapped GPU)
    - Any nested FAISS structure
    """



    # 🔥 ALWAYS convert to CPU before saving
    if isinstance(index, faiss.GpuIndex):
        index = faiss.index_gpu_to_cpu(index)
    else:
        # ⚠️ Important: handles wrapped GPU index (IndexPreTransform case)
        try:
            index = faiss.index_gpu_to_cpu(index)
        except:
            pass  # already CPU
    if clone:
        index = faiss.clone_index(index)
    return index  



def save_faiss_local(index, directory: str, filename: str):
    """
    Save FAISS index safely (supports GPU + wrappers)
    """
    index = _prepare_index_for_saving(index)
    filename = _normalize_faiss_filename(filename)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)

    # 🔥 ALWAYS convert to CPU before saving
    index = _prepare_index_for_saving(index, clone=True)

    faiss.write_index(index, path)
    print(f"Saved index to {path}")

    return path



import tempfile
import faiss
from huggingface_hub import upload_file,hf_hub_download


def save_faiss_hf(index, repo_id: str, filename: str):
    """
    Save FAISS index to HuggingFace Hub dataset repo.
    """
    index = _prepare_index_for_saving(index)
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


import faiss

def get_faiss_min_points_per_centroid():
    clus = faiss.Clustering(1, 1)  # dummy dims
    return clus.min_points_per_centroid

def get_printable_index(index):
    # Nếu là IndexIDMap/IndexIDMap2, lấy lõi bên trong trước
    if hasattr(index, "index"):
        actual_index = index.index
    else:
        actual_index = index
        
    # Ép kiểu xuống lớp con thực sự (IVF, PQ, Flat...)
    return faiss.downcast_index(actual_index)



