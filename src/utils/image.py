"""Image resizing helpers shared by all methods and the app."""
import numpy as np
from PIL import Image


def resize_long(image: np.ndarray, side: int, resample=Image.BILINEAR) -> np.ndarray:
    """Shrink an array so its long side is at most `side`, keeping aspect ratio."""
    img = Image.fromarray(image)
    img.thumbnail((side, side), resample)
    return np.asarray(img)


def resize_to(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a label mask to an exact size with nearest-neighbour sampling."""
    return np.asarray(Image.fromarray(mask).resize((width, height), Image.NEAREST))
