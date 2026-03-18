import faiss
import numpy as np
from .base import FaissIndex

class IndexHNSW(FaissIndex):

    def __init__(
        self,
        dimension=384,
        M=32,
        efConstruction=200,
        vectornormalize=True
    ):
        super().__init__(vectornormalize=vectornormalize)

        self.dimension = dimension
        self.M = M
        self.efConstruction = efConstruction

        self._create_index()

    def _create_index(self):

        if self.vectornormalize:
            metric = faiss.METRIC_INNER_PRODUCT
        else:
            metric = faiss.METRIC_L2

        base  = faiss.IndexHNSWFlat(
            self.dimension,
            self.M,
            metric
        )

        base.hnsw.efConstruction = self.efConstruction
        self.index = faiss.IndexIDMap2(base)