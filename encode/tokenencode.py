from .model import encodeModel
from typing import List, Union, Optional
import numpy as np
class tokenencodeModel(encodeModel):
    def __init__(
        self,
        model, **kwargs
    ):
        """
        token embedding wrapper.
        Args:
            model: encodeModel instance,model must have tokenencode method that returns token-level embeddings
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
            **kwargs: additional args passed to model.textencode()
            output: np.ndarray of embeddings of each token in the document, shape (documents,num_tokens, embedding_dim)
        """
        return self.model.tokenencode(documents, **kwargs)