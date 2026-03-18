import faiss
import numpy as np
from typing import Optional

from .utils import get_faiss_min_points_per_centroid            
from .base import FaissIndex

class IndexIVFSQ(FaissIndex):

    def __init__(
        self,
        dimension=384,
        nlist=None,
        qtype=faiss.ScalarQuantizer.QT_8bit,
        vectornormalize=True,
        min_points_per_centroid=None
    ):
        super().__init__(vectornormalize=vectornormalize)

        self.dimension = dimension
        self.nlist = nlist
        self.qtype = qtype

        self.min_points_per_centroid = (
            min_points_per_centroid
            if min_points_per_centroid is not None
            else get_faiss_min_points_per_centroid()
        )

        self._create_index()

    def _create_index(self):

        if self.vectornormalize:
            quantizer = faiss.IndexFlatIP(self.dimension)
            metric = faiss.METRIC_INNER_PRODUCT
        else:
            quantizer = faiss.IndexFlatL2(self.dimension)
            metric = faiss.METRIC_L2

        nlist = self.nlist if self.nlist is not None else 1

        self.index = faiss.IndexIVFScalarQuantizer(
            quantizer,
            self.dimension,
            nlist,
            self.qtype,
            metric
        )

        self._ensure_cp()

    def train(self, vectors: np.ndarray):

        if self.vectornormalize:
            faiss.normalize_L2(vectors)

        if self.nlist is None:
            self.nlist = max(
                int(len(vectors) / self.min_points_per_centroid), 1
            )
            print(f"[INFO] Auto nlist = {self.nlist}")
            self._create_index()

        if not self.index.is_trained:
            self.index.train(vectors.astype("float32"))

    def _ensure_cp(self):
        try:
            if hasattr(self.index, "cp"):
                self.index.cp.min_points_per_centroid = self.min_points_per_centroid
        except:
            pass