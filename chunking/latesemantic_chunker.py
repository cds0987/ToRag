from .base_chunker import BaseChunker
from .chunk import chunk
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Callable, Sequence, List, Tuple
from nltk.tokenize import sent_tokenize
from collections import defaultdict
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from .utils import ClusterStrategy

class LateSemanticChunker(BaseChunker):

    def __init__(
        self,
        encode,
        tokenizer,
        cluster_model=None,
        threshold: float = 0.75,
        max_chunk_size: int = 5,
        min_sentences: int = 2,
        max_sentences: int = 15,
    ):
        self.encode = encode
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.cluster_model = ClusterStrategy(cluster_model)
        self.max_chunk_size = max_chunk_size
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences

    def similarity(self, a, b):
        return float(cosine_similarity([a], [b])[0][0])

    def _sentence_embeddings(self, token_embeddings, spans):

        sent_embeds = []

        for start, end in spans:

            if end <= start:
                sent_embed = token_embeddings[start]
            else:
                sent_embed = token_embeddings[start:end].mean(axis=0)

            sent_embeds.append(sent_embed)

        return np.array(sent_embeds)

    def _sentence_char_spans(self, text, sentences):

        spans = []
        offset = 0

        for sent in sentences:

            start = text.find(sent, offset)

            if start == -1:
                raise ValueError(f"Sentence not found: {sent}")

            end = start + len(sent)

            spans.append((start, end))
            offset = end

        return spans

    def _sentence_token_spans(self, offsets, sent_char_spans):

        token_spans = []

        for s_start, s_end in sent_char_spans:

            start_token = None
            end_token = None

            for i, (tok_start, tok_end) in enumerate(offsets):

                if tok_end <= s_start:
                    continue

                if tok_start >= s_end:
                    break

                if start_token is None:
                    start_token = i

                end_token = i + 1

            if start_token is None:
                start_token = end_token = 0

            token_spans.append((start_token, end_token))

        return token_spans

    def preprocess_document(self, doc):

        token_embeddings = self.encode([doc])[0]

        sentences = sent_tokenize(doc)

        sent_char_spans = self._sentence_char_spans(doc, sentences)

        tokenized = self.tokenizer(
            doc,
            return_offsets_mapping=True,
            add_special_tokens=False
        )

        offsets = tokenized["offset_mapping"]

        token_spans = self._sentence_token_spans(
            offsets,
            sent_char_spans
        )

        max_tokens = token_embeddings.shape[0]

        valid_pairs = [
            (sent, span)
            for sent, span in zip(sentences, token_spans)
            if span[0] < max_tokens
        ]

        if not valid_pairs:
            return None

        sentences, token_spans = zip(*valid_pairs)

        sentences = list(sentences)
        token_spans = list(token_spans)

        sent_embeddings = self._sentence_embeddings(
            token_embeddings,
            token_spans
        )

        return sent_embeddings, sentences

    def clustering(self, sent_embeddings, sentences,**kwargs):

        n_sent = len(sentences)

        n_clusters = max(1, round(n_sent / self.max_chunk_size)) if kwargs.get("n_clusters") is None else kwargs["n_clusters"]
        labels = self.cluster_model.fit_predict(sent_embeddings,n_clusters = n_clusters)

        clusters = defaultdict(list)

        for idx, label in enumerate(labels):
            clusters[label].append((idx, sentences[idx]))

        sorted_clusters = sorted(
            clusters.items(),
            key=lambda x: min(i for i, _ in x[1])
        )

        ordered_sentences = []

        for _, items in sorted_clusters:

            items = sorted(items, key=lambda x: x[0])

            for idx, sent in items:
                ordered_sentences.append(sent)

        return ordered_sentences

    def split(self, documents,**kwargs):

        for doc_id, doc in enumerate(documents):

            result = self.preprocess_document(doc)

            if result is None:
                continue

            sent_embeddings, sentences = result

            if len(sent_embeddings) < 2:

                yield chunk(
                    doc_id=doc_id,
                    chunk_id=0,
                    text=" ".join(sentences),
                    metadata={"type": "late_semantic"}
                )
                continue

            ordered_sentences = self.clustering(
                sent_embeddings,
                sentences,
                **kwargs
            )

            current_chunk = []
            chunk_id = 0

            for sentence in ordered_sentences:

                current_chunk.append(sentence)

                if len(current_chunk) >= self.max_chunk_size:

                    yield chunk(
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        text=". ".join(current_chunk),
                        metadata={
                            "type": "late_semantic_cluster",
                            "cluster_chunksize": len(current_chunk)
                        }
                    )

                    chunk_id += 1
                    current_chunk = []

            if current_chunk:

                yield chunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=". ".join(current_chunk),
                    metadata={
                        "type": "late_semantic_cluster",
                        "cluster_chunksize": len(current_chunk)
                    }
                )
