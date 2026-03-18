import faiss
import numpy as np
from .base import FaissIndexIVF

class IndexIVFSQ(FaissIndexIVF):

    def __init__(self, qtype=faiss.ScalarQuantizer.QT_8bit, **kwargs):
        super().__init__(**kwargs)
        self.qtype = qtype

    def _build_ivf(self, quantizer, metric, nlist):
        return faiss.IndexIVFScalarQuantizer(
            quantizer,
            self.dimension,
            nlist,
            self.qtype,
            metric
        )