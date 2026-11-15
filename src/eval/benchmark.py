"""Efficiency measurements: runtime per image, parameter count, training time."""
import time
from contextlib import contextmanager

import numpy as np


def count_params(model) -> float:
    """Return trainable parameters in millions, or 0 for non-torch models."""
    params = getattr(model, "parameters", None)
    if params is None:
        return 0.0
    return sum(p.numel() for p in params() if p.requires_grad) / 1e6


def _sync(device: str) -> None:
    """Wait for queued CUDA work so timings are accurate."""
    if device.startswith("cuda"):
        import torch
        torch.cuda.synchronize()


def runtime_ms(predict, model, images: list[np.ndarray], device: str = "cpu", warmup: int = 2) -> float:
    """Return mean predict() wall time per image in milliseconds."""
    for img in images[:warmup]:
        predict(model, img)
    _sync(device)
    start = time.perf_counter()
    for img in images:
        predict(model, img)
    _sync(device)
    return 1000 * (time.perf_counter() - start) / len(images)


@contextmanager
def timer():
    """Context manager yielding a dict whose 'minutes' key is set on exit."""
    out = {}
    start = time.perf_counter()
    yield out
    out["minutes"] = (time.perf_counter() - start) / 60
