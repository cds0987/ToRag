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
    
    
from .base import FaissIndex

def faissIndexTogpu(faissIndex:FaissIndex, device: int = 0, res=None ):
   faissIndex.index,faissIndex.gpures   = FaissGPUUtils.to_gpu(faissIndex.index, device, res)
   
   
def faissIndexTocpu(faissIndex:FaissIndex):
   faissIndex.index = FaissGPUUtils.to_cpu(faissIndex.index)
   faissIndex.gpures = None