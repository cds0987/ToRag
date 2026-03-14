from abc import ABC, abstractmethod

from chunking.chunk import chunk
from typing import List


class BaseChunker(ABC):

    @abstractmethod
    def split(self, documents: List[str]) -> List[chunk]:
        pass
