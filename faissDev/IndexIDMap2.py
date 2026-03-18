import faiss
import numpy as np
import gc
from typing import Optional


class IndexIDMapAdapter:
    """
    Safe, memory-efficient adapter for FAISS ID mapping.

    Fixes:
    - No full dataset allocation
    - No unnecessary full index clone
    - Batch reconstruction
    - GPU-safe (optional CPU fallback)
    """

    # -------------------------
    # unwrap helper
    # -------------------------
    @staticmethod
    def _unwrap_index(index):
        while hasattr(index, "index"):
            index = index.index
        return faiss.downcast_index(index)

    # -------------------------
    # check support
    # -------------------------
    @staticmethod
    def supports_id(index) -> bool:
        base = IndexIDMapAdapter._unwrap_index(index)
        return hasattr(base, "add_with_ids")

    # -------------------------
    # check if already IDMap2
    # -------------------------
    @staticmethod
    def is_idmap(index) -> bool:
        return isinstance(index, faiss.IndexIDMap2)

    # -------------------------
    # wrap (non-destructive)
    # -------------------------
    @staticmethod
    def wrap(index):
        if IndexIDMapAdapter.is_idmap(index):
            return index

        base = IndexIDMapAdapter._unwrap_index(index)

        # IVF, HNSW, Flat usually already support IDs
        if hasattr(base, "add_with_ids"):
            return index

        return faiss.IndexIDMap2(index)

    # -------------------------
    # memory-safe rebuild
    # -------------------------
    @staticmethod
    def rebuild(
        index,
        ids: Optional[np.ndarray] = None,
        batch_size: int = 10000,
        to_cpu: bool = True
    ):
        """
        Memory-safe rebuild into IndexIDMap2.

        Args:
            index: FAISS index
            ids: optional external IDs
            batch_size: chunk size for reconstruction
            to_cpu: move GPU index → CPU before rebuild (recommended)

        Key improvements:
            - No full xb allocation
            - No full clone of populated index
            - Streaming add_with_ids
        """

        # -------------------------
        # move GPU → CPU (important)
        # -------------------------
        if to_cpu:
            try:
                index = faiss.index_gpu_to_cpu(index)
            except Exception:
                pass  # already CPU

        base = IndexIDMapAdapter._unwrap_index(index)

        if not hasattr(base, "reconstruct_n"):
            raise RuntimeError("Index does not support reconstruct_n")

        ntotal = index.ntotal

        if ntotal == 0:
            return faiss.IndexIDMap2(index)

        # -------------------------
        # resolve IDs
        # -------------------------
        if ids is not None:
            ids = np.asarray(ids, dtype="int64")
            if len(ids) != ntotal:
                raise ValueError("ids size must match index.ntotal")

        else:
            if isinstance(index, faiss.IndexIDMap2):
                ids = faiss.vector_to_array(index.id_map)
            else:
                ids = np.arange(ntotal, dtype="int64")

        # -------------------------
        # clone EMPTY structure only
        # -------------------------
        base_clone = faiss.clone_index(base)
        base_clone.reset()

        wrapped = faiss.IndexIDMap2(base_clone)

        # -------------------------
        # streaming reconstruction
        # -------------------------
        for i in range(0, ntotal, batch_size):
            n = min(batch_size, ntotal - i)

            xb = base.reconstruct_n(i, n)

            wrapped.add_with_ids(
                xb.astype("float32"),
                ids[i:i + n]
            )

            # free memory early
            del xb
            gc.collect()

        return wrapped

    # -------------------------
    # normalize entrypoint
    # -------------------------
    @staticmethod
    def ensure(index, force_rebuild=False, **kwargs):
        if IndexIDMapAdapter.is_idmap(index):
            return index

        if not force_rebuild:
            return IndexIDMapAdapter.wrap(index)

        return IndexIDMapAdapter.rebuild(index, **kwargs)