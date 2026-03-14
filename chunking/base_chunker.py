from abc import ABC, abstractmethod

from .chunk import chunk
from typing import List


class BaseChunker(ABC):

    @abstractmethod
    def split(self, documents: List[str]) -> List[chunk]:
        pass
