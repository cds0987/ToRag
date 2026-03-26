from .base import Retrieval

import numpy as np
from typing import List, Dict

class FaissRetrieval(Retrieval):
    """
    Abstract  Class for all Retrieval ,the core index from Faiss Library.

    """
    def __init__(self, index,datamanager):
        """
        Index is subclass from FaissIndex
        datamanager is any but have get_documents method
        """
        self.index = index
        self.datamanager = datamanager        
    def search(self,
               queries: np.ndarray,
               searchtype="textquery",
               index_kwargs=None,
               datamanager_kwargs=None
):
        index_kwargs = index_kwargs or {}
        datamanager_kwargs = datamanager_kwargs or {}
        I,D = self.index.search(queries, **index_kwargs)
        documents = self.datamanager.get_documents(I,D, **datamanager_kwargs)
        return documents