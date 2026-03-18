import faiss
import numpy as np
from typing import Optional

from .utils import get_faiss_min_points_per_centroid            
from .base import FaissIndexIVF

class IndexIVF_HNSW(FaissIndexIVF):

    def __init__(
        self,
        dimension=384,
        nlist = None,
        M=32,
        efConstruction=200,
        nprobe=16,
        min_points_per_centroid: Optional[int] = None,
        vectornormalize=True,
        directory=None,
        filename=None
    ):
        super().__init__(vectornormalize=vectornormalize)

        self.dimension = dimension
        self.nlist = nlist
        self.M = M
        self.efConstruction = efConstruction
        self.default_nprobe = nprobe
        self.min_points_per_centroid = (
            min_points_per_centroid
            if min_points_per_centroid is not None
            else get_faiss_min_points_per_centroid()
        )
        if directory and filename:
            self.load(directory, filename)
        else:
            self._create_index()

    # -------------------------
    # create index
    # -------------------------
    def _create_index(self):

        if self.vectornormalize:
            metric = faiss.METRIC_INNER_PRODUCT
        else:
            metric = faiss.METRIC_L2

        # 🔥 HNSW quantizer
        quantizer = faiss.IndexHNSWFlat(
            self.dimension,
            self.M,
            metric
        )
        quantizer.hnsw.efConstruction = self.efConstruction
        nlist = self.nlist if self.nlist is not None else 1
        # 🔥 IVF with HNSW quantizer
        index = faiss.IndexIVFFlat(
            quantizer,
            self.dimension,
            nlist,
            metric
        )
        # wrap with ID map
        self.index = faiss.IndexIDMap2(index)