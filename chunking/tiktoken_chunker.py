from .base_chunker import BaseChunker
from .chunk import chunk
import tiktoken


class TikTokenChunker(BaseChunker):

    def __init__(self, chunk_size=200, overlap=50, model="gpt-4"):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.step = chunk_size - overlap
        self.encoding = tiktoken.encoding_for_model(model)

    def split(self, documents):
        for doc_id, doc in enumerate(documents):

            tokens = self.encoding.encode(doc)
            chunk_id = 0

            for start in range(0, len(tokens), self.step):

                end = start + self.chunk_size
                token_slice = tokens[start:end]

                if not token_slice:
                    break

                text = self.encoding.decode(token_slice)

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
