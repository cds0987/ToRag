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
    # load,freeze it now later separately work on it
    # -------------------------
    def load(self, directory=None, filename=None, **kwargs):

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
            
            
#Ivf subclass
from .utils import get_faiss_min_points_per_centroid
import faiss
import numpy as np
from typing import Optional


class FaissIndexIVF(FaissIndex):

    def __init__(
        self,
        dimension=384,
        nlist=None,
        metric="ip",   # "ip" or "l2"
        vectornormalize=True,
        min_points_per_centroid=None,
    ):
        super().__init__(vectornormalize=vectornormalize)

        self.dimension = dimension
        self.nlist = nlist
        self.metric = metric

        self.min_points_per_centroid = (
            min_points_per_centroid
            if min_points_per_centroid is not None
            else get_faiss_min_points_per_centroid()
        )

        

    # -------------------------
    # metric + quantizer
    # -------------------------
    def _get_metric(self):
        return faiss.METRIC_INNER_PRODUCT if self.metric == "ip" else faiss.METRIC_L2

    def _create_quantizer(self):
        metric = self._get_metric()

        if metric == faiss.METRIC_INNER_PRODUCT:
            return faiss.IndexFlatIP(self.dimension)
        return faiss.IndexFlatL2(self.dimension)

    # -------------------------
    # FACTORY (override this)
    # -------------------------
    def _build_ivf(self, quantizer, metric, nlist):
        raise NotImplementedError

    # -------------------------
    # main creator
    # -------------------------
    def _create_index(self):
        if self.index is not None:
            return  # already created
        metric = self._get_metric()
        quantizer = self._create_quantizer()

        nlist = self.nlist if self.nlist is not None else 1

        base = self._build_ivf(quantizer, metric, nlist)
        self.index = base

    # -------------------------
    # train (shared)
    # -------------------------
    def train(self, vectors: np.ndarray):
        if self.vectornormalize:
            faiss.normalize_L2(vectors)

        if self.nlist is None:
            self.nlist = max(int(len(vectors) / self.min_points_per_centroid), 1)
            print(f"[INFO] Auto nlist = {self.nlist}")
        self._create_index()

        base = self._unwrap_index(self.index)

        if hasattr(base, "cp"):
            base.cp.min_points_per_centroid = self.min_points_per_centroid

        if not base.is_trained:
            base.train(vectors.astype("float32"))