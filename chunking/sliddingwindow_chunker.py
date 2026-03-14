from .base_chunker import BaseChunker
from .chunk import chunk

class SlidingWindowChunker(BaseChunker):

    def __init__(self, chunk_size=200, overlap=50):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.step = chunk_size - overlap

    def split(self, documents):
        for doc_id, doc in enumerate(documents):

            tokens = doc.split()  # simple tokenizer
            chunk_id = 0

            for start in range(0, len(tokens), self.step):

                end = start + self.chunk_size
                chunk_tokens = tokens[start:end]

                if not chunk_tokens:
                    break

                text = " ".join(chunk_tokens)

                yield chunk(
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        text=text,
                        metadata={
                            "start_token": start,
                            "end_token": min(end, len(tokens))
                        }
                    )
                

                chunk_id += 1

                if end >= len(tokens):
                    break
