import faiss       
from .base import FaissIndexIVF
from .registry import CLASS_REGISTRY

class IndexIVFPQ(FaissIndexIVF):

    def __init__(self, m=8, nbits=8, **kwargs):
        super().__init__(**kwargs)
        self.m = m
        self.nbits = nbits

    def _build_ivf(self, quantizer, metric, nlist):
        return faiss.IndexIVFPQ(
            quantizer,
            quantizer.d,
            nlist,
            self.m,
            self.nbits,
            metric
        )

CLASS_REGISTRY['IndexIVFPQ'] = IndexIVFPQ
