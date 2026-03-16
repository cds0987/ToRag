from typing import Callable, Iterable, List, Sequence,Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .base_chunker import BaseChunker
from .chunk import chunk
from nltk.tokenize import sent_tokenize

# Type alias for embedding function
EncodeFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]


class SemanticChunker(BaseChunker):

    def __init__(self, encode: EncodeFn, threshold: float = 0.75) -> None:
        """
        Args
        ----
        encode:
            Embedding function that maps:
            Sequence[str] -> embeddings

        threshold:
            Cosine similarity threshold for semantic split
        """
        self.encode = encode
        self.threshold = threshold
        

    def _split_sentences(self, text: str) -> List[str]:
        """
        Basic sentence splitter.
        """
        return sent_tokenize(text)

    def similarity(
        self,
        a: Sequence[float],
        b: Sequence[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        """
        return float(cosine_similarity([a], [b])[0][0])

    def _encode_documents(
        self,
        documents: Sequence[str]
    ) -> Tuple[List[List[str]], np.ndarray]:
        """
        Split all documents into sentences and encode them in one batch.

        Returns
        -------
        sentences_per_doc:
            List of sentence lists per document

        embeddings:
            Embeddings for ALL sentences
        """

        sentences_per_doc: List[List[str]] = []
        all_sentences: List[str] = []

        for doc in documents:
            sentences = self._split_sentences(doc)
            sentences_per_doc.append(sentences)
            all_sentences.extend(sentences)

        if not all_sentences:
            return sentences_per_doc, np.empty((0, 0))

        embeddings = np.asarray(self.encode(all_sentences))

        return sentences_per_doc, embeddings

    def split(self, documents: Sequence[str]) -> Iterable[chunk]:

        sentences_per_doc, embeddings = self._encode_documents(documents)

        global_index = 0

        for doc_id, sentences in enumerate(sentences_per_doc):

            if not sentences:
                continue

            current_chunk: List[str] = []
            chunk_id = 0

            for i, sentence in enumerate(sentences):

                current_chunk.append(sentence)

                if i == len(sentences) - 1:

                    yield chunk(
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        text=". ".join(current_chunk),
                        metadata={"type": "semantic"}
                    )

                    global_index += 1
                    break

                sim = self.similarity(
                    embeddings[global_index],
                    embeddings[global_index + 1]
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

                global_index += 1



class MeanCumulativeSemanticChunker(SemanticChunker):
   def split(self, documents: Sequence[str]) -> Iterable[chunk]:

    sentences_per_doc, embeddings = self._encode_documents(documents)

    global_index = 0

    for doc_id, sentences in enumerate(sentences_per_doc):

        if not sentences:
            continue

        current_chunk: List[str] = []
        current_embeddings: List[np.ndarray] = []

        chunk_id = 0

        for i, sentence in enumerate(sentences):

            current_chunk.append(sentence)
            current_embeddings.append(embeddings[global_index])

            # last sentence → flush
            if i == len(sentences) - 1:

                yield chunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=". ".join(current_chunk),
                    metadata={"type": "semantic"}
                )

                global_index += 1
                break

            # compute cumulative embedding
            chunk_embedding = np.mean(current_embeddings, axis=0)

            next_embedding = embeddings[global_index + 1]

            sim = self.similarity(
                chunk_embedding,
                next_embedding
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
                current_embeddings = []

            global_index += 1



from typing import Iterable, Sequence, List
import numpy as np


class CumulativeEcdSemanticChunker(SemanticChunker):

    def __init__(
        self,
        encode,
        threshold: float = 0.75
    ):
        super().__init__(encode, threshold)

    def split(self, documents: Sequence[str]) -> Iterable[chunk]:

        sentences_per_doc, embeddings = self._encode_documents(documents)

        global_index = 0

        for doc_id, sentences in enumerate(sentences_per_doc):

            if not sentences:
                continue

            current_chunk: List[str] = []
            chunk_id = 0

            for i, sentence in enumerate(sentences):

                current_chunk.append(sentence)

                # flush last sentence
                if i == len(sentences) - 1:

                    yield chunk(
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        text=". ".join(current_chunk),
                        metadata={"type": "semantic"}
                    )

                    global_index += 1
                    break

                # cumulative encoding of current chunk
                chunk_text = ". ".join(current_chunk)
                chunk_embedding = np.asarray(
                    self.encode([chunk_text])
                )[0]

                next_embedding = embeddings[global_index + 1]

                sim = self.similarity(chunk_embedding, next_embedding)

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

                global_index += 1


from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from collections import defaultdict
from .utils import ClusterStrategy

class ClusterSemanticChunker(SemanticChunker):

    def __init__(
        self,
        encode: EncodeFn,
        cluster_model=None,
        max_chunk_size: int = 5
    ):
        super().__init__(encode)
        self.cluster_model = ClusterStrategy(cluster_model)
        self.max_chunk_size = max_chunk_size


    def split(self, documents: Sequence[str], **kwargs) -> Iterable[chunk]:

        sentences_per_doc, embeddings = self._encode_documents(documents)

        global_index = 0

        for doc_id, sentences in enumerate(sentences_per_doc):

            if not sentences:
                continue

            n_sent = len(sentences)

            # slice embeddings belonging to this document
            doc_embeddings = embeddings[global_index: global_index + n_sent]

            global_index += n_sent

            n_clusters = max(1, n_sent // self.max_chunk_size) if kwargs.get("n_clusters") is None else kwargs["n_clusters"]
            labels = self.cluster_model.fit_predict(doc_embeddings,n_clusters = n_clusters)

            clusters = defaultdict(list)

            for idx, label in enumerate(labels):
                clusters[label].append((idx, sentences[idx]))

            chunk_id = 0

            # keep document order
            sorted_clusters = sorted(
                clusters.items(),
                key=lambda x: min(i for i, _ in x[1])
            )

            for label, items in sorted_clusters:

                items.sort(key=lambda x: x[0])

                text = ". ".join(sentence for _, sentence in items)

                yield chunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=text,
                    metadata={
                        "type": "cluster_semantic",
                        "cluster": int(label)
                    }
                )

                chunk_id += 1



