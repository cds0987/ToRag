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



import faiss
class FaissGPUUtils:

    # -------------------------
    # internal unwrap
    # -------------------------
    @staticmethod
    def unwrap_index(index):
        while hasattr(index, "index"):
            index = index.index
        return faiss.downcast_index(index)

    # -------------------------
    # check GPU compatibility
    # -------------------------
    @staticmethod
    def is_gpu_supported(index) -> bool:
        base = FaissGPUUtils.unwrap_index(index)

        # ❗ HNSW not supported
        if hasattr(base, "quantizer") and isinstance(base.quantizer, faiss.IndexHNSWFlat):
            return False

        return True

    # -------------------------
    # create GPU resource
    # -------------------------
    @staticmethod
    def create_gpu_resources():
        return faiss.StandardGpuResources()

    # -------------------------
    # CPU → GPU
    # -------------------------
    @staticmethod
    def to_gpu(cpu_index, device: int = 0, res=None):
        if isinstance(device, str):
            if device == "gpu":
                device = 0
            else:
                raise ValueError(f"Invalid device: {device}")

        device = int(device)

        if isinstance(cpu_index, faiss.GpuIndex):
            return cpu_index  # already GPU

        if not FaissGPUUtils.is_gpu_supported(cpu_index):
            raise RuntimeError("Index type not supported on GPU")

        if res is None:
            res = FaissGPUUtils.create_gpu_resources()

        return faiss.index_cpu_to_gpu(res, device, cpu_index), res

    # -------------------------
    # GPU → CPU
    # -------------------------
    @staticmethod
    def to_cpu(index):
        if isinstance(index, faiss.GpuIndex):
            return faiss.index_gpu_to_cpu(index)
        return index

    # -------------------------
    # clone CPU → GPU (sync)
    # -------------------------
    @staticmethod
    def clone_to_gpu(cpu_index, device: int = 0, res=None):
        return FaissGPUUtils.to_gpu(cpu_index, device, res)

    # -------------------------
    # clone GPU → CPU
    # -------------------------
    @staticmethod
    def clone_to_cpu(gpu_index):
        return FaissGPUUtils.to_cpu(gpu_index)

    # -------------------------
# check if index is on GPU
# -------------------------
    @staticmethod
    def is_gpu_index(index) -> bool:
      index = FaissGPUUtils.unwrap_index(index)
      return isinstance(index, faiss.GpuIndex)
    
    
from ToRag.faissDev.base import FaissIndex

def faissIndexTogpu(faissIndex:FaissIndex, device: int = 0, res=None ):
   faissIndex.index,faissIndex.gpures   = FaissGPUUtils.to_gpu(faissIndex.index)
def faissIndexTocpu(faissIndex:FaissIndex):
   faissIndex.index = FaissGPUUtils.to_cpu(faissIndex.index)
   faissIndex.gpures = None