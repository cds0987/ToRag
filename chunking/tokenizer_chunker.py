from .base_chunker import BaseChunker
from .chunk import chunk

class TokenizerChunker(BaseChunker):

    def __init__(self, tokenizer, chunk_size=200, overlap=50):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.step = chunk_size - overlap

    def _encode(self, text):
        # HuggingFace tokenizer returns list with add_special_tokens flag
        if hasattr(self.tokenizer, "encode"):
            try:
                return self.tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                return self.tokenizer.encode(text)

        raise ValueError("Tokenizer must support encode()")

    def _decode(self, tokens):
        if hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(tokens)

        raise ValueError("Tokenizer must support decode()")

    def split(self, documents):

        for doc_id, doc in enumerate(documents):

            tokens = self._encode(doc)
            chunk_id = 0

            for start in range(0, len(tokens), self.step):

                end = start + self.chunk_size
                token_slice = tokens[start:end]

                if not token_slice:
                    break

                yield chunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=self._decode(token_slice),
                    metadata={
                        "start_token": start,
                        "end_token": min(end, len(tokens)),
                        "token_count": len(token_slice)
                    }
                )

                chunk_id += 1

                if end >= len(tokens):
                    break
