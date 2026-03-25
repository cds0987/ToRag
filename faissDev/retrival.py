from ToRag.retriever.base import Retrieval

import numpy as np
from typing import List, Dict
# Assuming you use a common library like SentenceTransformers
# from sentence_transformers import SentenceTransformer

class FaissRetrieval(Retrieval):
    """
    Abstract  Class for all Retrieval ,the core index from Faiss Library.

    """
    def __init__(self, index,datamanager,encoder):
        """
        Index is subclass from FaissIndex
        Encoder is any but have textencode method
        datamanager is any but have get_documents method
        """
        self.index = index
        self.datamanager = datamanager
        self.encoder = encoder
    def encode_textquery(self, texts, **kwargs) -> np.ndarray:
        """Private helper to convert text to a vector."""
        embeddings = self.encoder.textencode(
        texts,**kwargs
    )
        return embeddings        
    def search(self,
               queries,
               searchtype="textquery",
               encoder_kwargs=None,
               index_kwargs=None,
               datamanager_kwargs=None
):
        encoder_kwargs = encoder_kwargs or {}
        index_kwargs = index_kwargs or {}
        datamanager_kwargs = datamanager_kwargs or {}
        if searchtype == 'textquery':
          embeddings = self.encode_textquery(queries,**encoder_kwargs)
        I,D = self.index.search(embeddings, **index_kwargs)
        documents = self.datamanager.get_documents(I,D, **datamanager_kwargs)
        return documents

