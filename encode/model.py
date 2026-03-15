from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np
class encodeModel(ABC):
    def __init__(self, model,*args, **kwargs):
        """
        model: model name or model instance
        kwargs: additional args passed to model loading function
        """
        self.model = model
    def textencode(self,*args, **kwargs):
        pass
    def tokenencode(
        self,
        text: str,
        *args,
        **kwargs
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Encode text and return token embeddings.

        Returns
        -------
        token_embeddings:
            shape (num_tokens, dim)

        offsets:
            list[(start_char, end_char)] per token
        """
        pass