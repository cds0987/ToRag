import numpy as np
import logging
import torch
from typing import List, Union, Optional, Any
from sentence_transformers import SentenceTransformer
from PIL import Image

from .model import encodeModel

logger = logging.getLogger(__name__)


class SentenceTransformerModel(encodeModel):
    def __init__(
        self,
        model: Union[str, SentenceTransformer],
        device: Optional[str] = None,
        normalize: bool = False,
        default_batch_size: int = 32,
        **kwargs
    ):
        """
        Extended SentenceTransformer wrapper with multi-modal support.

        Args:
            model: model name or instance
            device: "cuda", "cpu", etc.
            normalize: normalize embeddings by default
            default_batch_size: default batch size
        """
        if isinstance(model, str):
            self.model = SentenceTransformer(model, device=device, **kwargs)
        else:
            self.model = model

        self.device = device
        self.normalize = normalize
        self.default_batch_size = default_batch_size

    # -------------------------
    # internal helper
    # -------------------------
    def _encode(
        self,
        inputs: Any,
        batch_size: Optional[int] = None,
        normalize: Optional[bool] = None,
        **kwargs
    ) -> Optional[np.ndarray]:

        batch_size = batch_size or self.default_batch_size
        normalize = normalize if normalize is not None else self.normalize

        try:
            with torch.inference_mode():
                emb = self.model.encode(
                    inputs,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=normalize,
                    **kwargs
                )
            return emb

        except Exception as e:
            logger.error(f"Encoding error: {e}")
            return None

        finally:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    # -------------------------
    # text encoding
    # -------------------------
    def textencode(
        self,
        documents: Union[str, List[str]],
        **kwargs
    ) -> Optional[np.ndarray]:

        if isinstance(documents, str):
            documents = [documents]

        return self._encode(documents, **kwargs)

    # -------------------------
    # image encoding (CLIP only)
    # -------------------------
    def imgencode(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        **kwargs
    ) -> Optional[np.ndarray]:
        """
        Supports:
        - PIL images
        - image paths
        """

        if not isinstance(images, list):
            images = [images]

        processed_images = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            processed_images.append(img)

        return self._encode(processed_images, **kwargs)

    # -------------------------
    # unified encode
    # -------------------------
    def encode(
        self,
        inputs: Union[str, List[str], Image.Image, List[Image.Image]],
        modality: Optional[str] = None,
        **kwargs
    ) -> Optional[np.ndarray]:
        """
        Smart dispatcher.

        Args:
            inputs: text or images
            modality: "text" | "image" | None (auto-detect)
        """

        # explicit override
        if modality == "text":
            return self.textencode(inputs, **kwargs)
        if modality == "image":
            return self.imgencode(inputs, **kwargs)

        # auto detect
        if isinstance(inputs, (str, list)):
            if isinstance(inputs, list) and len(inputs) > 0:
                if isinstance(inputs[0], (Image.Image)):
                    return self.imgencode(inputs, **kwargs)
                elif isinstance(inputs[0], str):
                    return self.textencode(inputs, **kwargs)

            elif isinstance(inputs, str):
                return self.textencode(inputs, **kwargs)

        if isinstance(inputs, Image.Image):
            return self.imgencode(inputs, **kwargs)

        raise ValueError("Unsupported input type for encode()")

    # -------------------------
    # utility
    # -------------------------
    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def to(self, device: str):
        self.model.to(device)
        self.device = device