import faiss
from .base import FaissIndexIVF
from typing import Optional


class IndexIVFResidualQuantizer(FaissIndexIVF):

    def __init__(
        self,
        M: int = 8,          # residual stages
        nbits: int = 8,      # bits per stage
        **kwargs             # 🔥 catch ALL shared args (dimension, pca_dim, etc.)
    ):
        super().__init__(**kwargs)

        # private / specific params
        self.M = M
        self.nbits = nbits

    # -------------------------
    # IVF + Residual Quantizer
    # -------------------------
    def _build_ivf(self, quantizer, metric, nlist):

        index = faiss.IndexIVFResidualQuantizer(
            quantizer,
            quantizer.d,   # ✅ correct dimension
            nlist,
            self.M,
            self.nbits,
            metric
        )

        # recommended
        index.by_residual = True

        return index