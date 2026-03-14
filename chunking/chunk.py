from dataclasses import dataclass

@dataclass
class chunk:
    doc_id: int
    chunk_id: int
    text: str
    metadata: dict
