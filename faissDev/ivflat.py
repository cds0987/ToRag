import faiss
import numpy as np
from typing import Optional

from .utils import get_faiss_min_points_per_centroid            
from .base import FaissIndexIVF

class IndexIVFFlat(FaissIndexIVF):

    def _build_ivf(self, quantizer, metric, nlist):
        return faiss.IndexIVFFlat(
            quantizer,
            self.dimension,
            nlist,
            metric
        )

