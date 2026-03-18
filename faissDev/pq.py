import faiss
from faissDev.base import FaissIndex


class IndexPQ(FaissIndex):

    def __init__(
        self,
        dimension=384,
        m=8,
        nbits=8,
        vectornormalize=True
    ):
        super().__init__(vectornormalize=vectornormalize)

        self.dimension = dimension
        self.m = m
        self.nbits = nbits

        self._create_index()

    def _create_index(self):

        if self.vectornormalize:
            metric = faiss.METRIC_INNER_PRODUCT
        else:
            metric = faiss.METRIC_L2

        base = faiss.IndexPQ(
            self.dimension,
            self.m,
            self.nbits,
            metric
        )

        self.index = faiss.IndexIDMap2(base)