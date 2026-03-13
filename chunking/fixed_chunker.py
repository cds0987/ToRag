from .base_chunker import BaseChunker


class FixedChunker(BaseChunker):

    def __init__(self, chunk_size=500, overlap=50):

        self.chunk_size = chunk_size
        self.overlap = overlap


    def chunk(self, documents):

        for doc in documents:

            start = 0
            length = len(doc)

            while start < length:

                end = start + self.chunk_size

                yield doc[start:end]

                start = end - self.overlap
