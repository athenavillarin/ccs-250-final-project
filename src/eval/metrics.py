"""Binary segmentation metrics: mIoU, Dice, and pixel accuracy."""
import numpy as np


def confusion(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Return the 2x2 confusion matrix [[tn, fp], [fn, tp]] for binary masks."""
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    idx = 2 * gt.astype(bool).ravel() + pred.astype(bool).ravel()
    return np.bincount(idx, minlength=4).reshape(2, 2)


def scores(cm: np.ndarray) -> dict:
    """Compute mIoU (%), flood Dice, and pixel accuracy (%) from a confusion matrix."""
    (tn, fp), (fn, tp) = cm.astype(np.float64)
    iou_flood = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    iou_bg = tn / (tn + fp + fn) if tn + fp + fn else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
    return {
        "miou": float(100 * (iou_flood + iou_bg) / 2),
        "iou_flood": float(100 * iou_flood),
        "dice": float(dice),
        "pixel_acc": float(100 * (tp + tn) / cm.sum()),
    }


def evaluate(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Score one predicted mask against its ground truth."""
    return scores(confusion(pred, gt))
