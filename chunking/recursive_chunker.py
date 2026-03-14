from .base_chunker import BaseChunker
from .chunk import chunk
import re
class RecursiveChunker(BaseChunker):

    def __init__(
        self,
        chunk_size=500,
        overlap=50,
        separators=None
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or [
    r"\n\n",
    r"\n",
    r"(?<=[.!?])\s+",
    r"\s+",
    ""
]


    def split(self, documents):

        for doc_id, doc in enumerate(documents):
            chunks = self._recursive_split(doc, self.separators)
            chunks = self._apply_overlap(chunks)

            for chunk_id, chunk in enumerate(chunks):
                yield chunk(
                doc_id=doc_id,
                chunk_id=chunk_id,
                text=chunk,
                metadata={
                    "doc_id": doc_id,
                    "chunk_id": chunk_id
                }
            )
    def _word_count(self, text):
       return len(text.split())
    def _recursive_split(self, text, separators):

        if self._word_count(text) <= self.chunk_size:
            return [text.strip()]

        if not separators:
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        sep = separators[0]

        # fallback to char split
        if sep == "":
            return [
                text[i:i+self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        parts = re.split(sep, text) if sep else list(text)


        results = []
        buffer = ""

        for part in parts:
            part = part.strip()

            if not part:
                continue

            candidate = (buffer + " " + part).strip() if buffer else part

            if self._word_count(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    results.append(buffer)
                buffer = ""

                if self._word_count(part) > self.chunk_size:
                    results.extend(
                        self._recursive_split(part, separators[1:])
                    )
                else:
                    buffer = part

        if buffer:
            results.append(buffer)

        return results

    def _apply_overlap(self, chunks):

      if self.overlap <= 0:
        return chunks
      results = []
      for i, chunk in enumerate(chunks):

        if i == 0:
            results.append(chunk)
            continue

        prev = results[-1]

        prev_words = prev.split()
        overlap_words = prev_words[-self.overlap:]
        overlap_text = " ".join(overlap_words)

        results.append(overlap_text + " " + chunk)
      return results