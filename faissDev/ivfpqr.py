import faiss
from .base import FaissIndexIVF


class IndexIVFPQR(FaissIndexIVF):

    def __init__(self, m=16, nbits=8, m_refine=16, nbits_refine=8, **kwargs):
        super().__init__(**kwargs)

        self.m = m
        self.nbits = nbits
        self.m_refine = m_refine
        self.nbits_refine = nbits_refine

    def _build_ivf(self, quantizer, metric, nlist):
        return faiss.IndexIVFPQR(
            quantizer,
            self.dimension,
            nlist,
            self.m,
            self.nbits,
            self.m_refine,
            self.nbits_refine
        )