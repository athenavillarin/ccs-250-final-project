"""SegFormer-B0 fine-tuned for flood segmentation (stub until fine-tuning lands)."""
import numpy as np


def load(checkpoint_path: str | None = None, device: str = "cpu") -> object:
    """Return a ready-to-use model or parameter state."""
    return None


def predict(model, image: np.ndarray) -> np.ndarray:
    """Return an HxW uint8 {0,1} flood mask at the input image size."""
    return np.zeros(image.shape[:2], dtype=np.uint8)
