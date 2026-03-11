from llama_cpp import Llama
import logging
from pprint import pformat

logger = logging.getLogger(__name__)

def loadencoder(**kwargs):
    logger.info("Loading LlamaCpp model with configuration:\n%s", pformat(kwargs))
    kwargs['embedding'] = True
    model = Llama.from_pretrained(**kwargs)
    return model



from typing import List, Union, Optional
import numpy as np
from .model import encodeModel
class textencode(encodeModel):
    def __init__(
        self,
        model_name: str, **kwargs
    ):
        """
        Professional text embedding wrapper.
        Args:
            model: model name or SentenceTransformer instance
            device: cpu / cuda / mps
        """

        super().__init__(model_name, **kwargs)
        self.model = loadencoder(**kwargs)
    def encode(
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
        return embeddings