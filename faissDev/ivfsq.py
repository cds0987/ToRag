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
            quantizer.d,   # 🔥 critical fix
            nlist,
            self.qtype,
            metric
        )

from .registry import CLASS_REGISTRY
CLASS_REGISTRY['IndexIVFSQ'] = IndexIVFSQ