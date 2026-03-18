from typing import Optional
import faiss
import numpy as np
from ToRag.vectorIndex.base import VectorIndex
from .utils import (
    save_faiss_hf,
    load_faiss_hf,
    load_faiss_local,
    save_faiss_local
    )
class FaissIndex(VectorIndex):

    def __init__(self, *args, **kwargs):
        self.index = None
        self.vectornormalize = kwargs.get("vectornormalize", False)

    # -------------------------
    # internal helper
    # -------------------------
    def _unwrap_index(self, index):
        while True:
            if hasattr(index, "index"):
                index = index.index
            else:
                break
        index = faiss.downcast_index(index)
        return index
    # -------------------------
    # ntotal
    # -------------------------
    def ntotal(self):
        return self.index.ntotal
        # -------------------------
    # set_nprobe
    # -------------------------
    def set_nprobe(self, nprobe: int):
        ivf_index = self._unwrap_index(self.index)
        if hasattr(ivf_index, "nprobe"):
           ivf_index.nprobe = nprobe
        # -------------------------
    # -------------------------
    # train
    # -------------------------
    def train(self, vectors: np.ndarray):

        if self.index is None:
            raise ValueError("Index is not initialized")

        if self.vectornormalize:
            faiss.normalize_L2(vectors)
        base = self._unwrap_index(self.index)
        if not base.is_trained:
            base.train(vectors.astype("float32"))

    # -------------------------
    # add
    # -------------------------
    def add(self, ids: np.ndarray, vectors: np.ndarray):

        if self.index is None:
            raise ValueError("Index is not initialized")

        if vectors.shape[0] != ids.shape[0]:
            raise ValueError("ids and vectors size mismatch")

        if self.vectornormalize:
            faiss.normalize_L2(vectors)

        base = self._unwrap_index(self.index)

        if hasattr(base, "is_trained") and not base.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")

        ids = ids.astype("int64")

        self.index.add_with_ids(vectors.astype("float32"), ids)

    # -------------------------
    # search
    # -------------------------
    def search(self, query_vector: np.ndarray, top_k: int = 5, nprobe: Optional[int] = None):

        if self.index is None:
            raise ValueError("Index is not initialized")

        if nprobe is not None:
            self.set_nprobe(nprobe)

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        if self.vectornormalize:
            faiss.normalize_L2(query_vector)

        D, I = self.index.search(query_vector.astype("float32"), top_k)

        return I, D  # 🔥 raw output only

    # -------------------------
    # delete
    # -------------------------
    def delete(self, ids: np.ndarray):

        if self.index is None:
            raise ValueError("Index is not initialized")

        ids = ids.astype("int64")

        return self.index.remove_ids(ids)

    # -------------------------
    # save
    # -------------------------
    def save(self, index=None, directory=None, filename=None, **kwargs):

        local = kwargs.get("local", False)

        if index is None:
            index = self.index

        if local:
            save_faiss_local(index, directory, filename)
        else:
            save_faiss_hf(index, directory, filename)

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