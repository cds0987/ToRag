import faiss          
from .base import FaissIndexIVF

class IndexIVFFlat(FaissIndexIVF):

    def _build_ivf(self, quantizer, metric, nlist):
        return faiss.IndexIVFFlat(
            quantizer,
            quantizer.d,   # ✅ critical fix
            nlist,
            metric
        )

from .registry import CLASS_REGISTRY
CLASS_REGISTRY['IndexIVFFlat'] = IndexIVFFlat