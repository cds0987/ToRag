from typing import List, Union, Tuple, Optional
from PIL import Image
import torch
from ToRag.Utils.image import load_image
def build_qwen_vl_inputs(
    tokenizer,
    images: List[Union[str, Image.Image]],
    texts: List[str],
    device="cuda",
    max_length=2048,

    # 🔥 new options
    resize: bool = False,
    image_size: Tuple[int, int] = (512, 512),
    keep_ratio: bool = False,
):
    """
    Build batched inputs for Qwen-VL style models.

    Args:
        tokenizer: Qwen processor/tokenizer
        images: list of image paths or PIL images
        texts: list of text prompts
        resize: whether to resize images
        image_size: target size (width, height)
        keep_ratio: keep aspect ratio if resizing
    """

    assert len(images) == len(texts), "images and texts must match"

    # ✅ Step 1: load + optional resize
    pil_images = [
        load_image(img, resize=resize, size=image_size, keep_ratio=keep_ratio)
        for img in images
    ]

    # ✅ Step 2: build chat-style prompts
    prompts = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ],
            }
        ]
        for text in texts
    ]

    # ✅ Step 3: convert to model text format
    processed_texts = [
        tokenizer.apply_chat_template(p, tokenize=False)
        for p in prompts
    ]

    # ✅ Step 4: tokenize
    inputs = tokenizer(
        text=processed_texts,
        images=pil_images,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return {k: v.to(device) for k, v in inputs.items()}


import torch
from typing import List, Union
from PIL import Image
from ToRag.Utils.iterior import create_iterior
from .model import encodeModel

class QwenVLEncoder(encodeModel):

    def __init__(self, model, tokenizer, device="cuda", batch_size=4, normalize=True):
        super().__init__(model)
        model.to(device)
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize

        self.hidden_states = []

        def hook_fn(module, input, output):
            self.hidden_states.append(output)

        self.hook = self.model.model.language_model.layers[-1].register_forward_hook(hook_fn)

    # =========================
    # Core
    # =========================
    def _clear_hook_buffer(self):
        self.hidden_states.clear()

    def _forward(self, inputs):
        self._clear_hook_buffer()
        with torch.inference_mode():
            self.model(**inputs)
        return self.hidden_states.pop()

    def _mean_pool(self, hidden, mask):
        emb = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        if self.normalize:
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb

    def _get_dummy_image(self):
        return Image.new("RGB", (224, 224), (128, 128, 128))

    def _prepare_inputs(self, images, texts):
        if images is None:
            images = []
        if texts is None:
            texts = []

        if not isinstance(images, list):
            images = [images]
        if not isinstance(texts, list):
            texts = [texts]

        n = max(len(images), len(texts))

        if len(images) == 0:
            images = [self._get_dummy_image()] * n
        elif len(images) < n:
            images = images + [self._get_dummy_image()] * (n - len(images))

        if len(texts) == 0:
            texts = [""] * n
        elif len(texts) < n:
            texts = texts + [""] * (n - len(texts))

        return images, texts

    def _build_iterator(self, length, show_progress_bar):
        iterator = range(0, length, self.batch_size)
        return create_iterior(iterator) if show_progress_bar else iterator

    # =========================
    # 🔹 Low-level APIs
    # =========================

    def encodeWhole(self, images, texts, show_progress_bar=True, **kwargs):
        """
        Sentence-level embedding → (N, dim)
        """
        images, texts = self._prepare_inputs(images, texts)

        all_embeddings = []
        iterator = self._build_iterator(len(images), show_progress_bar)

        for i in iterator:
            inputs = build_qwen_vl_inputs(
                self.tokenizer,
                images[i:i+self.batch_size],
                texts[i:i+self.batch_size],
                device=self.device,
                **kwargs
            )

            hidden = self._forward(inputs)
            emb = self._mean_pool(hidden, inputs["attention_mask"])

            all_embeddings.append(emb.cpu())

            del inputs, hidden, emb
            torch.cuda.empty_cache()

        return torch.cat(all_embeddings, dim=0)

    def encodeToken(self, images, texts, show_progress_bar=True, **kwargs):
        """
        Token-level embedding → List[(B, seq_len, dim)]
        """
        images, texts = self._prepare_inputs(images, texts)

        all_tokens = []
        iterator = self._build_iterator(len(images), show_progress_bar)

        for i in iterator:
            inputs = build_qwen_vl_inputs(
                self.tokenizer,
                images[i:i+self.batch_size],
                texts[i:i+self.batch_size],
                device=self.device,
                **kwargs
            )

            hidden = self._forward(inputs)
            all_tokens.append(hidden.cpu())

            del inputs, hidden
            torch.cuda.empty_cache()

        return all_tokens

    # =========================
    # 🔹 High-level interface
    # =========================

    def encode(
        self,
        images: Union[List, Image.Image, None] = None,
        texts: Union[List[str], str, None] = None,
        output: str = "sentence",   # 🔥 key interface
        show_progress_bar=True,
        **kwargs
    ):
        """
        Unified interface

        output:
            - "sentence" → (N, dim)
            - "token"    → List[(B, seq_len, dim)]
        """

        if output == "sentence":
            return self.encodeWhole(images, texts, show_progress_bar, **kwargs)

        elif output == "token":
            return self.encodeToken(images, texts, show_progress_bar, **kwargs)

        else:
            raise ValueError(f"Unknown output type: {output}")

    # =========================
    # Cleanup
    # =========================
    def close(self):
        self.hook.remove()