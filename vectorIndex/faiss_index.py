from ast import Dict, List
from email.mime import base

import faiss
import numpy as np
from .base import VectorIndex
from .utils import save_faiss_hf, load_faiss_hf, load_faiss_local, save_faiss_local

import os

class FaissIndex(VectorIndex):

    def __init__(self, *args, **kwargs):
        self.index = None
    def delete(self, ids: List[str]):
        pass
    def search(self, query_vector: np.ndarray, top_k: int = 5):
        pass
    def add(self, ids: List[str], vectors: np.ndarray, metadata: List[Dict] = None):
        pass
    # -------------------------
    # save
    # -------------------------
    def save(self, index=None, directory=None, filename=None, path=None, **kwargs):

        local = kwargs.get("local", False)

        if path is not None:
            directory, filename = self._split_path(path)

        if index is None:
            index = self.index

        if local:
            save_faiss_local(index, directory, filename)
        else:
            save_faiss_hf(index, directory, filename)

    # -------------------------
    # set_nprobe
    # -------------------------
    def set_nprobe(self, nprobe: int):
        base = self.index.index
        base.nprobe = nprobe
    # -------------------------
    # get ntotal
    # -------------------------
    def ntotal(self):
       return self.index.ntotal
    # -------------------------
    # load
    # -------------------------
    def load(self, directory=None, filename=None, path=None):

        try:
            index = load_faiss_local(directory, filename)

        except FileNotFoundError:
            try:
                index = load_faiss_hf(directory, filename)

            except Exception as e:
                raise FileNotFoundError(
                    f"Index not found in both local and HuggingFace Hub: {e}"
                )

        index = faiss.IndexIDMap2(index) if not isinstance(index, faiss.IndexIDMap2) else index
        base = index.index if isinstance(index, faiss.IndexIDMap2) else index

        # restore config from index
        self.dimension = base.d
        self.nlist = base.nlist
        self.m = base.pq.M
        self.nbits = base.pq.nbits

        self.trained = base.is_trained
        self.index = index

        # detect metric
        if base.metric_type == faiss.METRIC_INNER_PRODUCT:
            self.vectornormalize = True
        else:
            self.vectornormalize = False
            
class IndexIVFPQ(FaissIndex):

    def __init__(self,
        dimension: int = 384,
        nlist: int = 100,
        m: int = 8,
        nbits: int = 8,vectornormalize = True, 
        directory=None,filename = None):
        
        self.index = None
        self.dimension = dimension
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.trained = False

        # id mapping
        self.id_map = {}
        self.rev_id_map = {}
        self.next_id = 0
        self.vectornormalize = vectornormalize
        if directory  and filename:
            self.load(directory, filename)
        else:
            self._create_index()
            
    # -------------------------
    # create IVF-PQ index
    # -------------------------
    def _create_index(self):
        if self.vectornormalize:
           quantizer = faiss.IndexFlatIP(self.dimension)
        else:
            quantizer = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexIVFPQ(
            quantizer,
            self.dimension,
            self.nlist,
            self.m,
            self.nbits, 
            faiss.METRIC_INNER_PRODUCT if self.vectornormalize else faiss.METRIC_L2
        )

        self.index = faiss.IndexIDMap2(index)
        
    def train(self, vectors: np.ndarray):
        if self.vectornormalize:
           faiss.normalize_L2(vectors)
        base = self.index.index  # unwrap IDMap2
        if not base.is_trained:
           base.train(vectors)
           self.trained = True