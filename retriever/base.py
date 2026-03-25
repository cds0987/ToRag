from abc import ABC, abstractmethod
from typing import List, Any

class Retrieval(ABC):
    """
    Abstract Base Class for all Retrieval from all components.
    """
    def __init__(self, *args, **kwargs):
        pass
    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Any]:
        """
        Search the index for the top k relevant items.
        
        Args:
            query (str): The search input.
            k (int): Number of results to return.
        """
        pass