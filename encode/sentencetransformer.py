from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logger = logging.getLogger(__name__)
import torch
class TextEncoder:
    def __init__(
        self,
        model: Union[str, SentenceTransformer],
        device: Optional[str] = None,
    ):
        """
        Professional text embedding wrapper.
        Args:
            model: model name or SentenceTransformer instance
            device: cpu / cuda / mps
        """

        if isinstance(model, str):
            logger.info(f"Loading embedding model: {model}")
            self.model = SentenceTransformer(model, device=device)
        else:
            self.model = model
        self.model.eval()

    def encode(
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

        if isinstance(documents, str):
            documents = [documents]

        with torch.no_grad():
            embeddings = self.model.encode(
            documents,
            **kwargs
        )

        return embeddings