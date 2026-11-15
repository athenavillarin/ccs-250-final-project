"""SegFormer-B0 (ImageNet-pretrained MiT-B0 encoder) fine-tuned for flood segmentation."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerImageProcessor
from transformers.utils import logging

from src.utils.paths import CHECKPOINT_DIR
from src.utils.train import predict_mask

PRETRAINED = "nvidia/mit-b0"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "pretrained_mixed_s42.pt"


class SegFormer(nn.Module):
    """SegFormer with a one-logit head, its own normalisation, and logits upsampled to the input size."""

    def __init__(self, name: str | None = PRETRAINED):
        """Load pretrained weights and processor statistics, or build an empty B0 when name is None."""
        super().__init__()
        if name is None:
            self.net = SegformerForSemanticSegmentation(SegformerConfig(num_labels=1))
            mean, std = torch.zeros(3), torch.ones(3)
        else:
            verbosity = logging.get_verbosity()
            logging.set_verbosity_error()
            self.net = SegformerForSemanticSegmentation.from_pretrained(name, num_labels=1)
            logging.set_verbosity(verbosity)
            proc = SegformerImageProcessor.from_pretrained(name)
            mean, std = torch.tensor(proc.image_mean), torch.tensor(proc.image_std)
        self.register_buffer("mean", mean.view(1, 3, 1, 1))
        self.register_buffer("std", std.view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a Bx3xHxW image in [0,1] to Bx1xHxW logits."""
        logits = self.net(pixel_values=(x - self.mean) / self.std).logits
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def load(checkpoint_path: str | None = None, device: str = "cpu") -> SegFormer:
    """Build SegFormer-B0 offline and load fine-tuned weights and normalisation statistics."""
    model = SegFormer(None)
    model.load_state_dict(torch.load(checkpoint_path or DEFAULT_CHECKPOINT, map_location=device))
    return model.to(device).eval()


def predict(model: SegFormer, image: np.ndarray) -> np.ndarray:
    """Return an HxW uint8 {0,1} flood mask at the input image size."""
    return predict_mask(model, image, str(next(model.parameters()).device))
