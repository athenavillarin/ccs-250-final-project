"""SegFormer-B0 (ImageNet-pretrained MiT-B0 encoder) fine-tuned for flood segmentation."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from transformers.utils import logging

from src.utils.paths import CHECKPOINT_DIR
from src.utils.train import predict_mask

PRETRAINED = "nvidia/mit-b0"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "pretrained_train_s42.pt"


class SegFormer(nn.Module):
    """SegFormer with a one-logit head, its own normalisation, and logits upsampled to the input size."""

    def __init__(self, name: str = PRETRAINED):
        """Load pretrained encoder weights and the matching image processor statistics."""
        super().__init__()
        verbosity = logging.get_verbosity()
        logging.set_verbosity_error()
        self.net = SegformerForSemanticSegmentation.from_pretrained(name, num_labels=1)
        logging.set_verbosity(verbosity)
        proc = SegformerImageProcessor.from_pretrained(name)
        self.register_buffer("mean", torch.tensor(proc.image_mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(proc.image_std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a Bx3xHxW image in [0,1] to Bx1xHxW logits."""
        logits = self.net(pixel_values=(x - self.mean) / self.std).logits
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def load(checkpoint_path: str | None = None, device: str = "cpu") -> SegFormer:
    """Build SegFormer-B0 and load fine-tuned weights."""
    model = SegFormer()
    model.load_state_dict(torch.load(checkpoint_path or DEFAULT_CHECKPOINT, map_location=device))
    return model.to(device).eval()


def predict(model: SegFormer, image: np.ndarray) -> np.ndarray:
    """Return an HxW uint8 {0,1} flood mask at the input image size."""
    return predict_mask(model, image, str(next(model.parameters()).device))
