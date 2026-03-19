import faiss       
from .base import FaissIndexIVF

class IndexIVF_HNSW(FaissIndexIVF):

    def __init__(self, M=32, efConstruction=200, **kwargs):
        super().__init__(**kwargs)
        self.M = M
        self.efConstruction = efConstruction

    def _create_quantizer(self):
        metric = self._get_metric()

        quantizer = faiss.IndexHNSWFlat(
            self.dimension,
            self.M,
            metric
        )
        quantizer.hnsw.efConstruction = self.efConstruction
        return quantizer

    def _build_ivf(self, quantizer, metric, nlist):
        return faiss.IndexIVFFlat(
            quantizer,
            self.dimension,
            nlist,
            metric
        )