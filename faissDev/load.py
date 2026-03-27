from .registry import CLASS_REGISTRY
from .utils import (
    load_faiss_hf,
    load_faiss_local
    )
from ToRag.Utils.loadHgface import load_json_hf
from ToRag.Utils.loadFileLocal import load_json_local
import inspect

def filter_kwargs(cls, metadata: dict):
    sig = inspect.signature(cls.__init__)
    valid_params = sig.parameters.keys()

    return {
        k: v for k, v in metadata.items()
        if k in valid_params
    }

def from_metadata(metadata: dict):
    cls_name = metadata.get("class")

    if cls_name not in CLASS_REGISTRY:
        raise ValueError(f"Unknown class: {cls_name}")

    cls = CLASS_REGISTRY[cls_name]

    kwargs = filter_kwargs(cls, metadata)

    obj = cls(**kwargs)

    return obj

    
from ToRag.faissDev.gpuUtils import faissIndexTogpu
def load_faiss_index(directoryidx = None, fileidx = None, 
                    directorymeta=None, filemeta=None, **kwargs):
    try:
        index = load_faiss_local(directoryidx, fileidx)
    except FileNotFoundError:
        try:
            index = load_faiss_hf(directoryidx, fileidx)
        except Exception as e:
            raise FileNotFoundError(
                f"Index not found in both local and HuggingFace Hub: {e}"
            )
    try:
        metadata = load_json_hf(directorymeta, filemeta)
    except FileNotFoundError:
        try:
            metadata = load_json_local(directorymeta, filemeta)
        except Exception as e:
            raise FileNotFoundError(
                f"Metadata not found in both local and HuggingFace Hub: {e}"
            )
    newfaissIndex = from_metadata(metadata)
    newfaissIndex.index = index
    checkgpu = metadata['gpu'] if 'gpu' in metadata else False
    if checkgpu:
        faissIndexTogpu(newfaissIndex, **kwargs)
    return newfaissIndex