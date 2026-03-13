import numpy as np
import logging
logger = logging.getLogger(__name__)
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional



from .model import encodeModel
class sentencetransformermodel(encodeModel):
    def __init__(
        self,
        model: Union[torch.nn.Module, SentenceTransformer], **kwargs
    ):
        """
        Professional text embedding wrapper.
        Args:
            model: torch.nn.Module format or SentenceTransformer instance
        """
        super().__init__(model, **kwargs)
    def textencode(
        self,
        documents: Union[str, List[str]],
        **kwargs,
    ) -> np.ndarray:
        """
        Encode documents into embeddings.
        Args:
            documents: text or list of texts
            **kwargs: additional args passed to model.encode(),similar with Sentence Transformer library
        """
        embeddings = None
        if isinstance(documents, str):
            documents = [documents]
        try:
          with torch.no_grad():
            embeddings = self.model.encode(
            documents,
            convert_to_numpy = True,
            **kwargs
        )
        except Exception as e:
            logger.error(f"Error encoding documents: {e}")
            pass
        torch.cuda.empty_cache()
        return embeddings
    
