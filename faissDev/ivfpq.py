import faiss
import numpy as np
from typing import Optional

from .utils import get_faiss_min_points_per_centroid            
from .base import FaissIndexIVF


class IndexIVFPQ(FaissIndexIVF):

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
