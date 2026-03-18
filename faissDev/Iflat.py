import faiss
import numpy as np
from .base import FaissIndex


    
class IndexFlat(FaissIndex):

    def __init__(self, dimension=384, vectornormalize=True, metric="ip"):
        """
        metric: "ip" | "l2"
        """
        super().__init__(vectornormalize=vectornormalize)

        self.dimension = dimension
        self.metric = metric.lower()

        self._validate_config()
        self._create_index()

    # -------------------------
    # validate
    # -------------------------
    def _validate_config(self):
        if self.metric not in ["ip", "l2"]:
            raise ValueError("metric must be 'ip' or 'l2'")

        # ❌ không normalize mà dùng IP → sai cosine
        if not self.vectornormalize and self.metric == "ip":
            raise ValueError(
                "IndexFlatIP requires vectornormalize=True (cosine similarity assumption)"
            )

    # -------------------------
    # create index
    # -------------------------
    def _create_index(self):

        if self.metric == "ip":
            base = faiss.IndexFlatIP(self.dimension)

        elif self.metric == "l2":
            base = faiss.IndexFlatL2(self.dimension)

        self.index = faiss.IndexIDMap2(base)