"""ONNX Runtime inference for the exported deep models, mirroring predict_mask without importing torch."""
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from src.utils.image import resize_long


class OnnxModel:
    """An ONNX Runtime session plus the input side and parameter count stored in the model metadata."""

    def __init__(self, path: str | Path):
        """Open the model on CPU, using FLOOD_THREADS intra-op threads when set."""
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("FLOOD_THREADS", 0))
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        self.side = int(meta["side"])
        self.params_m = float(meta["params_m"])


def load(checkpoint_path: str | Path, device: str = "cpu") -> OnnxModel:
    """Load an exported .onnx model; device is accepted for interface parity and always CPU."""
    return OnnxModel(checkpoint_path)


def predict(model: OnnxModel, image: np.ndarray) -> np.ndarray:
    """Return an HxW uint8 {0,1} flood mask at the input image size."""
    small = resize_long(image, model.side)
    h, w = small.shape[:2]
    x = np.pad(small.astype(np.float32) / 255, ((0, -h % 32), (0, -w % 32), (0, 0)), mode="reflect")
    logits = model.session.run(None, {"image": x.transpose(2, 0, 1)[None]})[0][0, 0, :h, :w]
    up = Image.fromarray(np.ascontiguousarray(logits)).resize((image.shape[1], image.shape[0]), Image.BILINEAR)
    return (np.asarray(up) > 0).astype(np.uint8)
