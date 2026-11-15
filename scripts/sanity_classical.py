"""Synthetic sanity check for the graph cut: a known two-region image with heavy noise."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.metrics import evaluate
from src.methods import classical
from src.utils.paths import RESULTS_DIR
from src.utils.seed import SEED
from src.utils.viz import SERIES, overlay, plt, save

H, W = 192, 256
WATER, LAND = (120, 100, 75), (70, 125, 60)
NOISE = 30
LAMBDAS = (0, 1, 2, 5, 10, 20)
SIGMAS = (15, 30, 60)


def scene() -> np.ndarray:
    """Ground-truth mask: a winding river plus a round pond."""
    y, x = np.mgrid[:H, :W]
    river = np.abs(y - (H / 2 + 30 * np.sin(x / 30))) < 22
    pond = (y - 40) ** 2 + (x - 200) ** 2 < 28**2
    return (river | pond).astype(np.uint8)


def render(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Paint muddy water and vegetation colours, then add Gaussian noise."""
    img = np.where(mask[..., None] == 1, WATER, LAND).astype(np.float32)
    return np.clip(img + rng.normal(0, NOISE, img.shape), 0, 255).astype(np.uint8)


def main() -> None:
    """Fit colour models on one noisy render, sweep lambda/sigma on another, save table and figures."""
    rng = np.random.default_rng(SEED)
    gt = scene()
    model = classical.fit([(render(gt, rng), gt)], k=3)
    test = render(gt, rng)

    rows = []
    for sigma in SIGMAS:
        for lam in LAMBDAS:
            model.update(lam=lam, sigma=sigma)
            rows.append({"sigma": sigma, "lambda": lam, **evaluate(classical.predict(model, test), gt)})
    df = pd.DataFrame(rows).round(3)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "sanity_classical.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for color, (sigma, g) in zip(SERIES, df.groupby("sigma")):
        ax.plot(g["lambda"], g["miou"], marker="o", color=color, label=f"σ = {sigma}")
    ax.set_xlabel("Smoothness weight λ (unitless)")
    ax.set_ylabel("mIoU (%)")
    ax.set_title("Graph cut on a synthetic noisy scene")
    ax.legend(title="Contrast scale (Lab units)")
    save(fig, RESULTS_DIR / "sanity_classical_sweep.png")

    best = df.loc[df["miou"].idxmax()]
    model.update(lam=0, sigma=best["sigma"])
    pixelwise = classical.predict(model, test)
    model.update(lam=best["lambda"])
    cut = classical.predict(model, test)
    panels = [
        ("Noisy input", test),
        ("Ground truth", overlay(test, gt)),
        (f"Pixel-wise (λ = 0)\nmIoU {evaluate(pixelwise, gt)['miou']:.1f}%", overlay(test, pixelwise)),
        (f"Graph cut (λ = {best['lambda']:g}, σ = {best['sigma']:g})\nmIoU {best['miou']:.1f}%", overlay(test, cut)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    save(fig, RESULTS_DIR / "sanity_classical.png")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
