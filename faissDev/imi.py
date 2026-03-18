from .base import FaissIndex
import faiss
import numpy as np
from typing import Optional


class IndexIMI(FaissIndex):

    def __init__(
        self,
        dimension=384,
        nbits=8,     # bits per sub-quantizer
        m=16,        # PQ subvectors
        nprobe=16,
        vectornormalize=True
    ):
        super().__init__(vectornormalize=vectornormalize)

        self.dimension = dimension
        self.nbits = nbits
        self.m = m
        self.default_nprobe = nprobe

        self.quantizer = None
        self.base = None

        self._create_index()

    # -------------------------
    # create index
    # -------------------------
    def _create_index(self):

        if self.vectornormalize:
            metric = faiss.METRIC_INNER_PRODUCT
        else:
            metric = faiss.METRIC_L2

        # 🔥 Multi-index quantizer (2-way split)
        self.quantizer = faiss.MultiIndexQuantizer(
            self.dimension,
            2,              # usually 2
            self.nbits
        )

        # total cells = (2^nbits)^2
        nlist = 2 ** (self.nbits * 2)

        # 🔥 IVF + PQ with IMI quantizer
        self.base = faiss.IndexIVFPQ(
            self.quantizer,
            self.dimension,
            nlist,
            self.m,
            8,              # PQ bits
            metric
        )

        self.base.nprobe = self.default_nprobe

        # 🔥 CRITICAL: prevent FAISS from retraining quantizer
        self.base.quantizer_trains_alone = True

        self.index = faiss.IndexIDMap2(self.base)

    # -------------------------
    # train (FIXED for IMI)
    # -------------------------
    def train(self, vectors: np.ndarray):

        if self.index is None:
            raise ValueError("Index is not initialized")

        if self.vectornormalize:
            faiss.normalize_L2(vectors)

        vectors = vectors.astype("float32")

        # 🔥 Step 1: train IMI quantizer manually
        if not self.quantizer.is_trained:
            self.quantizer.train(vectors)

        # 🔥 Step 2: train IVF-PQ WITHOUT touching quantizer
        if not self.base.is_trained:
            self.base.train(vectors)