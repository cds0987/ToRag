from .base_chunker import BaseChunker
from .chunk import chunk

class FixedChunker(BaseChunker):

    def __init__(self, chunk_size=500, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, documents):

        for doc_id, doc in enumerate(documents):

            words = doc.split()
            start = 0
            length = len(words)
            chunk_id = 0

            while start < length:

                end = start + self.chunk_size
                chunk_words = words[start:end]

                text = " ".join(chunk_words)

                yield chunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=text,
                    metadata={
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "start_word": start,
                        "end_word": min(end, length)
                    }
                )

                chunk_id += 1
                start = end - self.overlap

