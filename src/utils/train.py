"""Shared training and inference loop for the deep models (BCE + Dice, AdamW, early stopping)."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import TRAIN_SIDE
from src.eval.metrics import scores
from src.utils.image import resize_long


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy plus batch-level soft Dice, which stays well-defined for flood-free images."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    dice = 1 - (2 * (prob * target).sum() + 1) / (prob.sum() + target.sum() + 1)
    return bce + dice


def forward_padded(model, x: torch.Tensor) -> torch.Tensor:
    """Reflect-pad H and W to multiples of 32, run the model, and crop logits back."""
    h, w = x.shape[-2:]
    return model(F.pad(x, (0, -w % 32, 0, -h % 32), mode="reflect"))[..., :h, :w]


@torch.no_grad()
def evaluate_loader(model, loader, device: str) -> tuple[float, dict]:
    """BCE + Dice pooled over all pixels of the loader, and pooled segmentation scores."""
    model.eval()
    bce, pixels, inter, prob_sum, target_sum = 0.0, 0, 0.0, 0.0, 0.0
    cm = torch.zeros(2, 2, dtype=torch.long, device=device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.split(":")[0], enabled=device.startswith("cuda")):
            logits = forward_padded(model, x).float()
        prob = torch.sigmoid(logits)
        bce += F.binary_cross_entropy_with_logits(logits, y, reduction="sum").item()
        pixels += y.numel()
        inter += (prob * y).sum().item()
        prob_sum += prob.sum().item()
        target_sum += y.sum().item()
        cm += torch.bincount(2 * y.long().flatten() + (logits > 0).long().flatten(), minlength=4).reshape(2, 2)
    dice = 1 - (2 * inter + 1) / (prob_sum + target_sum + 1)
    return bce / pixels + dice, scores(cm.cpu().numpy())


def train(model, train_ds, val_ds, out_path: str | Path, epochs: int = 60, lr: float = 1e-3,
          weight_decay: float = 1e-4, batch_size: int = 8, patience: int = 10, device: str = "cuda",
          num_workers: int = 2) -> tuple[pd.DataFrame, float]:
    """Train with early stopping on val mIoU, saving the best weights; returns (history, minutes)."""
    model.to(device)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers, drop_last=True,
                          pin_memory=True, persistent_workers=num_workers > 0)
    val_dl = DataLoader(val_ds, 1, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = torch.amp.GradScaler(enabled=device.startswith("cuda"))
    history, best, stale = [], -1.0, 0
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.split(":")[0], enabled=device.startswith("cuda")):
                loss = bce_dice_loss(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * len(x)
        sched.step()
        val_loss, val = evaluate_loader(model, val_dl, device)
        row = {"epoch": epoch, "train_loss": running / (len(train_dl) * batch_size), "val_loss": val_loss,
               "val_miou": val["miou"], "val_dice": val["dice"], "lr": sched.get_last_lr()[0]}
        history.append(row)
        print(" ".join(f"{k}={v:.4g}" for k, v in row.items()), flush=True)
        if val["miou"] > best:
            best, stale = val["miou"], 0
            torch.save(model.state_dict(), out_path)
        else:
            stale += 1
            if stale >= patience:
                break
    minutes = (time.perf_counter() - start) / 60
    model.load_state_dict(torch.load(out_path, map_location=device))
    return pd.DataFrame(history), minutes


@torch.no_grad()
def predict_mask(model, image: np.ndarray, device: str) -> np.ndarray:
    """Run a logit model at TRAIN_SIDE and upsample logits to the input size before thresholding."""
    model.eval()
    small = resize_long(image, TRAIN_SIDE)
    x = torch.from_numpy(np.array(small)).permute(2, 0, 1)[None].float().div(255).to(device)
    logits = forward_padded(model, x)
    logits = F.interpolate(logits.float(), size=image.shape[:2], mode="bilinear", align_corners=False)
    return (logits[0, 0] > 0).to(torch.uint8).cpu().numpy()
