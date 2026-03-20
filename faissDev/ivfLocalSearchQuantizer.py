import faiss          
from .base import FaissIndexIVF
# -------------------------
# IVF + Local Search Quantizer
# -------------------------
class IndexIVFLocalSearchQuantizer(FaissIndexIVF):

    def __init__(
        self,
        M=8,
        nbits=8,
        **kwargs   # 🔥 important
    ):
        super().__init__(**kwargs)

        self.M = M
        self.nbits = nbits

    def _build_ivf(self, quantizer, metric, nlist):
        return faiss.IndexIVFLocalSearchQuantizer(
            quantizer,
            quantizer.d,   # ✅ critical fix
            nlist,
            self.M,
            self.nbits,
            metric
        )