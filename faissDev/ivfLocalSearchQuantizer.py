import faiss          
from .base import FaissIndexIVF
# -------------------------
# IVF + Local Search Quantizer
# -------------------------
class IndexIVFLocalSearchQuantizer(FaissIndexIVF):

    def __init__(
        self,
        dimension=384,
        nlist=None,
        M=8,
        nbits=8,
        metric="ip",
        vectornormalize=True,
        min_points_per_centroid=None,
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

    def _build_ivf(self, quantizer, metric, nlist):
        return faiss.IndexIVFLocalSearchQuantizer(
        quantizer,      # The coarse quantizer (IndexFlat)
        self.dimension,  # d
        nlist,          # nlist
        self.M,          # M (number of sub-quantizers)
        self.nbits,      # nbits
        metric          # metric
    )