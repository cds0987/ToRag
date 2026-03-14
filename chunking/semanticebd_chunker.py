from typing import Callable, Iterable, List, Sequence
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .base_chunker import BaseChunker
from .chunk import chunk


# Type alias for embedding function
EncodeFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]


class SemanticChunker(BaseChunker):

    def __init__(self, encode: EncodeFn, threshold: float = 0.75) -> None:
        """
        Args:
            encode: Callable embedding function that converts
                    List[str] -> embeddings (2D array-like)
            threshold: similarity threshold for semantic splitting
        """
        self.encode = encode
        self.threshold = threshold

    def _split_sentences(self, text: str) -> List[str]:
        """
        Basic sentence splitter.
        """
        return [s.strip() for s in text.split(".") if s.strip()]

    def similarity(
        self,
        a: Sequence[float],
        b: Sequence[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        """
        return float(cosine_similarity([a], [b])[0][0])

    def split(self, documents: Sequence[str]) -> Iterable[chunk]:

        for doc_id, doc in enumerate(documents):

            sentences: List[str] = self._split_sentences(doc)

            if not sentences:
                continue

            embeddings = np.asarray(self.encode(sentences))

            current_chunk: List[str] = []
            chunk_id: int = 0

            for i, sentence in enumerate(sentences):

                current_chunk.append(sentence)

                if i == len(sentences) - 1:
                    yield chunk(
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        text=". ".join(current_chunk),
                        metadata={"type": "semantic"}
                    )
                    break

                sim: float = self.similarity(
                    embeddings[i],
                    embeddings[i + 1]
                )

                if sim < self.threshold:

                    yield chunk(
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        text=". ".join(current_chunk),
                        metadata={
                            "type": "semantic",
                            "similarity_break": sim
                        }
                    )

                    chunk_id += 1
                    current_chunk = []
