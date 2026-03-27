from typing import List, Union, Tuple, Optional
from PIL import Image
def load_image(
    x,
    resize: bool = False,
    size: Tuple[int, int] = (512, 512),
    keep_ratio: bool = False,
):
    """Convert input to PIL.Image with optional resizing"""
    
    if isinstance(x, Image.Image):
        img = x.convert("RGB")
    elif isinstance(x, str):
        img = Image.open(x).convert("RGB")
    else:
        raise ValueError(f"Unsupported image type: {type(x)}")

    if resize:
        if keep_ratio:
            img.thumbnail(size, Image.BICUBIC)  # keeps aspect ratio
        else:
            img = img.resize(size, Image.BICUBIC)

    return img