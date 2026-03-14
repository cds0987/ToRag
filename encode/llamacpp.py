try:
  from llama_cpp import Llama
except ImportError:
  raise ImportError("llama_cpp library is required for llamacppModel. Please install it with `pip install llama-cpp-python` or never use llamacppModel else unexpected behavior.")
  Llama = None
import logging
from pprint import pformat

logger = logging.getLogger(__name__)
from typing import List, Union, Optional
import numpy as np
from .model import encodeModel
class llamacppModel(encodeModel):
    def __init__(
        self,
        model: Llama, **kwargs
    ):
        """
        Professional text embedding wrapper.
        Args:
            model: llama_cpp.llama.Llama model
        """
        super().__init__(model, **kwargs)
    def textencode(
        self,
        documents: Union[str, List[str]],
        **kwargs,
    ):
        """
        Encode documents into embeddings.
        Args:
            documents: single string or list of strings to encode
            **kwargs: additional args passed to model encoding function
        Returns:
            List of embeddings for each document
        """
        vectors = []
        if isinstance(documents, str):
            documents = [documents]
        try:
            for  doc in documents:
                emb = self.model.create_embedding(doc)
                vectors.append(emb["data"][0]["embedding"])
        except Exception as e:
            logger.error(f"Error encoding documents: {e}")
        embeddings = np.array(vectors).astype("float32")
        normalize = kwargs.get("normalize", True)
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
        return embeddings