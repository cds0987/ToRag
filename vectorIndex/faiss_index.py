from typing import List, Dict
import faiss
import numpy as np
from .base import VectorIndex
from .utils import save_faiss_hf, load_faiss_hf, load_faiss_local, save_faiss_local

import os

class FaissIndex(VectorIndex):

    def __init__(self, *args, **kwargs):
        self.index = None
        self.id_map = {}
        self.rev_id_map = {}
        self.next_id = 0
        self.vectornormalize = kwargs.get("vectornormalize", False)

    def delete(self, ids: List[str]):
        pass

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        pass

    # -------------------------
    # add
    # -------------------------
    def add(self, ids: List[str], vectors: np.ndarray, metadata: List[Dict] = None):

        if self.index is None:
            raise ValueError("Index is not initialized")

        # normalize nếu dùng cosine/IP
        if self.vectornormalize:
            faiss.normalize_L2(vectors)

        base = self.index.index if hasattr(self.index, "index") else self.index

        # chỉ check nếu index cần train
        if hasattr(base, "is_trained") and not base.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")

        # đảm bảo có IDMap2
        self._ensure_idmap2_safe()

        int_ids = []

        for sid in ids:
            if sid in self.id_map:
                int_id = self.id_map[sid]
            else:
                int_id = self.next_id
                self.id_map[sid] = int_id
                self.rev_id_map[int_id] = sid
                self.next_id += 1

            int_ids.append(int_id)

        int_ids = np.array(int_ids).astype("int64")

        # thêm vector vào index
        if hasattr(self.index, "add_with_ids"):
            self.index.add_with_ids(vectors, int_ids)
        else:
            # fallback (hiếm khi xảy ra)
            self.index.add(vectors)

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
    # _ensure_idmap2_safe
    # -------------------------
    def _ensure_idmap2_safe(self):

       if isinstance(self.index, faiss.IndexIDMap2):
          return

       if self.index.ntotal == 0:
          self.index = faiss.IndexIDMap2(self.index)
          return

       print("[WARNING] Rebuilding index to add IDMap2...")

       ntotal = self.index.ntotal
       vectors_old = self.index.reconstruct_n(0, ntotal)

       base = faiss.clone_index(self.index)
       base.reset()

       new_index = faiss.IndexIDMap2(base)

       old_ids = np.arange(ntotal).astype("int64")
       new_index.add_with_ids(vectors_old, old_ids)

       self.index = new_index

       # rebuild id map
       self.id_map = {}
       self.rev_id_map = {}

       for i in range(ntotal):
           sid = str(i)
           self.id_map[sid] = i
           self.rev_id_map[i] = sid

       self.next_id = ntotal

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

        self.index = index
        self._ensure_idmap2_safe()

        base = index.index if isinstance(index, faiss.IndexIDMap2) else index

        # restore config from index
        self.dimension = base.d
        self.nlist = base.nlist
        self.m = base.pq.M
        self.nbits = base.pq.nbits

        self.trained = base.is_trained

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