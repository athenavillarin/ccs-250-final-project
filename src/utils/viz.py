"""Shared figure style, method colours, and mask overlays for report plots."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHOD_COLORS = {"classical": "#2a78d6", "scratch": "#eb6834", "pretrained": "#1baf7a"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
FLOOD_RGB = np.array([42, 120, 214])
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "legend.frameon": False,
})


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend the flood colour over pixels where mask is 1."""
    out = image.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * FLOOD_RGB
    return out.astype(np.uint8)


def save(fig, path: str | Path) -> None:
    """Save a figure as PNG and close it."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_curves(history, title: str, path: str | Path) -> None:
    """Plot train/val loss and val mIoU per epoch, marking the best epoch."""
    best = history.loc[history["val_miou"].idxmax()]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    ax1.plot(history["epoch"], history["train_loss"], color=SERIES[0], label="Train")
    ax1.plot(history["epoch"], history["val_loss"], color=SERIES[1], label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE + Dice loss (unitless)")
    ax1.set_title("Loss")
    ax1.legend()
    ax2.plot(history["epoch"], history["val_miou"], color=SERIES[1])
    ax2.axvline(best["epoch"], color=INK_2, linestyle="--", linewidth=1)
    ax2.annotate(f"best {best['val_miou']:.1f}% @ epoch {int(best['epoch'])}", (best["epoch"], best["val_miou"]),
                 xytext=(6, -14), textcoords="offset points", color=INK_2, fontsize=9)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation mIoU (%)")
    ax2.set_title("Validation mIoU")
    fig.suptitle(title)
    save(fig, path)
