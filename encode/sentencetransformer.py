from typing import List, Union, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


def encode_text(
    documents: Union[str, List[str]],
    model: Union[str, SentenceTransformer],
    batch_size: int = 32,
    device: Optional[str] = None,
    normalize_embeddings: bool = True,
    show_progress_bar: bool = False,
    convert_to_numpy: bool = True,
    **kwargs
) -> np.ndarray:
    """
    Encode text into embeddings using SentenceTransformer.

    Args:
        documents: text or list of texts
        model: model name or SentenceTransformer instance
        batch_size: encoding batch size
        device: cpu / cuda / mps
        normalize_embeddings: normalize vectors for cosine similarity
        show_progress_bar: show encoding progress
        convert_to_numpy: return numpy array
        **kwargs: additional args passed to model.encode()

    Returns:
        np.ndarray: embeddings
    """

    if isinstance(documents, str):
        documents = [documents]

    if not isinstance(documents, list):
        raise TypeError("documents must be str or List[str]")

    if len(documents) == 0:
        raise ValueError("documents cannot be empty")

    # Load model if needed
    if isinstance(model, str):
        logger.info(f"Loading embedding model: {model}")
        model = SentenceTransformer(model, device=device)

    model.eval()

    embeddings = model.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=normalize_embeddings,
        convert_to_numpy=convert_to_numpy,
        **kwargs
    )

    return embeddings