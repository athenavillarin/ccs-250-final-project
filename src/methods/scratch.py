"""U-Net written from scratch and trained from random weights for flood segmentation."""
import numpy as np
import torch
from torch import nn

from src.utils.paths import CHECKPOINT_DIR
from src.utils.train import predict_mask

DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "scratch_train_s42.pt"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class DoubleConv(nn.Sequential):
    """Two 3x3 conv + batch norm + ReLU layers."""

    def __init__(self, cin: int, cout: int):
        """Build the two conv layers."""
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class Up(nn.Module):
    """Transposed-conv upsampling, skip concatenation, then DoubleConv."""

    def __init__(self, cin: int, cout: int):
        """Build the upsampling and fusion layers."""
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 2, stride=2)
        self.conv = DoubleConv(2 * cout, cout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample x and fuse it with the encoder skip feature."""
        return self.conv(torch.cat([self.up(x), skip], 1))


class UNet(nn.Module):
    """Four-level U-Net returning one flood logit per pixel at the input size."""

    def __init__(self, base: int = 32, dropout: float = 0.2):
        """Build encoder, bottleneck, and decoder with widths base * 2^level."""
        super().__init__()
        widths = [base * 2**i for i in range(5)]
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
        self.inc = DoubleConv(3, widths[0])
        self.downs = nn.ModuleList(nn.Sequential(nn.MaxPool2d(2), DoubleConv(a, b)) for a, b in zip(widths, widths[1:]))
        self.drop = nn.Dropout2d(dropout)
        self.ups = nn.ModuleList(Up(a, b) for a, b in zip(widths[::-1], widths[::-1][1:]))
        self.head = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a Bx3xHxW image in [0,1] (H, W divisible by 16) to Bx1xHxW logits."""
        x = self.inc((x - self.mean) / self.std)
        skips = [x]
        for down in self.downs:
            x = down(x)
            skips.append(x)
        x = self.drop(skips.pop())
        for up in self.ups:
            x = up(x, skips.pop())
        return self.head(x)


def load(checkpoint_path: str | None = None, device: str = "cpu") -> UNet:
    """Build the U-Net and load trained weights."""
    model = UNet()
    model.load_state_dict(torch.load(checkpoint_path or DEFAULT_CHECKPOINT, map_location=device))
    return model.to(device).eval()


def predict(model: UNet, image: np.ndarray) -> np.ndarray:
    """Return an HxW uint8 {0,1} flood mask at the input image size."""
    return predict_mask(model, image, str(next(model.parameters()).device))
