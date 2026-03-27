import faiss       
from .base import FaissIndexIVF

class IndexIVF_HNSW(FaissIndexIVF):

    def __init__(self, M=32, efConstruction=200, **kwargs):
        super().__init__(**kwargs)
        self.M = M
        self.efConstruction = efConstruction

    # 🔥 dimension-aware quantizer
    def _create_quantizer(self, dim=None):
        metric = self._get_metric()
        dim = dim if dim is not None else self.dimension

        quantizer = faiss.IndexHNSWFlat(
            dim,
            self.M,
            metric
        )
        quantizer.hnsw.efConstruction = self.efConstruction
        return quantizer

    # 🔥 dimension-safe IVF
    def _build_ivf(self, quantizer, metric, nlist):
        return faiss.IndexIVFFlat(
            quantizer,
            quantizer.d,   # 🔥 critical fix
            nlist,
            metric
        )

from .registry import CLASS_REGISTRY
CLASS_REGISTRY['IndexIVF_HNSW'] = IndexIVF_HNSW