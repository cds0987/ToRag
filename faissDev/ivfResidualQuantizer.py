import faiss
from .base import FaissIndexIVF
from typing import Optional


class IndexIVFResidualQuantizer(FaissIndexIVF):

    def __init__(
        self,
        dimension: int = 384,
        nlist: Optional[int] = None,
        M: int = 8,              # number of residual stages
        nbits: int = 8,          # bits per stage
        metric: str = "ip",      # "ip" or "l2"
        vectornormalize: bool = True,
        min_points_per_centroid: Optional[int] = None,
    ):
        self.M = M
        self.nbits = nbits

        super().__init__(
            dimension=dimension,
            nlist=nlist,
            metric=metric,
            vectornormalize=vectornormalize,
            min_points_per_centroid=min_points_per_centroid,
        )

    # -------------------------
    # IVF + Residual Quantizer (FIXED)
    # -------------------------
    def _build_ivf(self, quantizer, metric, nlist):

        index = faiss.IndexIVFResidualQuantizer(
            quantizer,
            self.dimension,
            nlist,
            self.M,        # ✅ correct
            self.nbits,    # ✅ correct
            metric
        )

        # 🔥 optional but recommended
        index.by_residual = True

        return index