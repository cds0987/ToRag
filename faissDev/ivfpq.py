import faiss
import numpy as np
from typing import Optional

from .utils import get_faiss_min_points_per_centroid            
from .base import FaissIndex


class IndexIVFPQ(FaissIndex):

    def __init__(
        self,
        dimension: int = 384,
        nlist: Optional[int] = None,
        m: int = 8,
        nbits: int = 8,
        vectornormalize: bool = True,
        min_points_per_centroid: Optional[int] = None,
        directory=None,
        filename=None
    ):
        """_summary_

        Args:
            dimension (int, optional): _description_. Defaults to 384.
            nlist (Optional[int], optional): _description_. Defaults to None.
            m (int, optional): _description_. Defaults to 8.
            nbits (int, optional): _description_. Defaults to 8.
            vectornormalize (bool, optional): _description_. Defaults to True.
            min_points_per_centroid (Optional[int], optional): _description_. Defaults to None.
            directory (_type_, optional): _description_. Defaults to None.
            filename (_type_, optional): _description_. Defaults to None.
        """
        super().__init__(vectornormalize=vectornormalize)

        self.dimension = dimension
        self.nlist = nlist
        self.m = m
        self.nbits = nbits

        self.trained = False

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
    # create IVF-PQ index
    # -------------------------
    def _create_index(self):

        if self.vectornormalize:
            quantizer = faiss.IndexFlatIP(self.dimension)
            metric = faiss.METRIC_INNER_PRODUCT
        else:
            quantizer = faiss.IndexFlatL2(self.dimension)
            metric = faiss.METRIC_L2

        nlist = self.nlist if self.nlist is not None else 1

        self.index = faiss.IndexIVFPQ(
            quantizer,
            self.dimension,
            nlist,
            self.m,
            self.nbits,
            metric
        )
        self._ensure_min_points_per_centroid()

    # -------------------------
    # train
    # -------------------------
    def train(self, vectors: np.ndarray):

        if self.vectornormalize:
            faiss.normalize_L2(vectors)

        n = vectors.shape[0]

        # -------------------------
        # auto compute nlist
        # -------------------------
        if self.nlist is None:
            self.nlist = max(int(n / self.min_points_per_centroid), 1)
            print(f"[INFO] Auto nlist = {self.nlist}")

            # rebuild index
            self._create_index()

        self._ensure_min_points_per_centroid()

        if not self.index.is_trained:
            self.index.train(vectors.astype("float32"))
            self.trained = True

    # -------------------------
    # helper: clustering param
    # -------------------------
    def _ensure_min_points_per_centroid(self):

        try:
            base = self._unwrap_index(self.index)

            if hasattr(base, "cp"):
                base.cp.min_points_per_centroid = self.min_points_per_centroid
            else:
                print("[WARNING] Index does not support clustering params")

        except Exception as e:
            print(f"[WARNING] Cannot set min_points_per_centroid: {e}")