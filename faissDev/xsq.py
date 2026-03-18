import faiss
from faissDev.base import FaissIndex

class IndexSQ(FaissIndex):

    def __init__(
        self,
        dimension=384,
        qtype=faiss.ScalarQuantizer.QT_8bit,
        vectornormalize=True
    ):
        super().__init__(vectornormalize=vectornormalize)

        self.dimension = dimension
        self.qtype = qtype

        self._create_index()

    # -------------------------
    # create index
    # -------------------------
    def _create_index(self):

        if self.vectornormalize:
            metric = faiss.METRIC_INNER_PRODUCT
        else:
            metric = faiss.METRIC_L2

        base = faiss.IndexScalarQuantizer(
            self.dimension,
            self.qtype,
            metric
        )

        self.index = faiss.IndexIDMap2(base)