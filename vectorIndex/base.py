from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class VectorIndex(ABC):

    @abstractmethod
    def add(self, ids: List[str], vectors: np.ndarray, metadata: List[Dict] = None):
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5):
        pass

    @abstractmethod
    def delete(self, ids: List[str]):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass
