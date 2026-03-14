from .base_chunker import BaseChunker
from .chunk import chunk



class HierarchicalChunker(BaseChunker):

    def __init__(self, tokenizer, chunk_size=200, overlap=50):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
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

            sections = doc.split("\n# ")  # simple section split

            for sec_id, section in enumerate(sections):

                paragraphs = section.split("\n\n")

                for para_id, para in enumerate(paragraphs):

                    tokens = self._encode(para)
                    chunk_id = 0

                    for start in range(0, len(tokens), self.step):

                        token_slice = tokens[start:start + self.chunk_size]

                        if not token_slice:
                            break

                        yield chunk(
                            doc_id=doc_id,
                            chunk_id=chunk_id,
                            text=self._decode(token_slice),
                            metadata={
                                'document_id': doc_id,
                                "section_id": sec_id,
                                "paragraph_id": para_id,
                                "start_token": start,
                                "token_count": len(token_slice)
                            }
                        )

                        chunk_id += 1