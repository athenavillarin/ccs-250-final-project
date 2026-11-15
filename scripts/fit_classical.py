"""Fit graph-cut feature models on a train split and tune lambda, sigma, and bias on a fixed val subset."""
import argparse
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import MIXES, load_pair, read_split
from src.eval.metrics import confusion, scores
from src.methods import classical
from src.utils.paths import CHECKPOINT_DIR, RESULTS_DIR
from src.utils.runs import log_run
from src.utils.seed import SEED
from src.utils.viz import SERIES, plt, save

LAMBDAS = (0, 5, 10, 20, 40, 80)
SIGMAS = (10, 20)
BIASES = (-2.0, -1.0, -0.5, 0.0, 0.5)
GRID = list(product(SIGMAS, BIASES, LAMBDAS))


def sweep_image(args) -> np.ndarray:
    """Confusion matrices for every (sigma, bias, lambda) on one val image."""
    model, split, image_id = args
    image, gt = load_pair(split, image_id)
    out = []
    for sigma, bias, lam in GRID:
        model.update(lam=lam, sigma=sigma, bias=bias)
        out.append(confusion(classical.predict(model, image), gt))
    return np.array(out)


def main() -> None:
    """Fit on a train split, grid-search on its val sets, save the best parameters, table, and plot."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="mixed", choices=("train", "mixed"))
    ap.add_argument("--val-per-set", type=int, default=40)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    start = time.perf_counter()
    model = classical.fit(load_pair(p, i) for p in MIXES.get(args.split, (args.split,)) for i in read_split(p))
    fit_min = (time.perf_counter() - start) / 60
    val_sets = MIXES.get(f"{args.split}_val", ("val",))
    jobs = [(model, p, i) for p in val_sets for i in random.Random(SEED).sample(read_split(p), args.val_per_set)]
    with ProcessPoolExecutor(args.workers) as pool:
        cms = sum(pool.map(sweep_image, jobs))
    rows = [{"sigma": s, "bias": b, "lambda": l, **scores(cm)} for (s, b, l), cm in zip(GRID, cms)]
    df = pd.DataFrame(rows).round(3)
    df.to_csv(RESULTS_DIR / f"classical_tuning_{args.split}.csv", index=False)
    best = df.loc[df["miou"].idxmax()]
    model.update(lam=float(best["lambda"]), sigma=float(best["sigma"]), bias=float(best["bias"]))
    classical.save(model, CHECKPOINT_DIR / f"classical_{args.split}.npz")
    log_run("classical", args.split, SEED, (time.perf_counter() - start) / 60, 0.0, float(best["miou"]))

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    at_sigma = df[df["sigma"] == best["sigma"]]
    for color, (bias, g) in zip(SERIES, at_sigma.groupby("bias")):
        ax.plot(g["lambda"], g["miou"], marker="o", color=color, label=f"b = {bias:g}")
    ax.set_xlabel("Smoothness weight λ (unitless)")
    ax.set_ylabel("Validation mIoU (%)")
    ax.set_title(f"Graph-cut tuning on {len(jobs)} {args.split} val images (σ = {best['sigma']:g})")
    ax.legend(title="Flood bias b (nats)")
    save(fig, RESULTS_DIR / f"classical_tuning_{args.split}.png")
    print(df.sort_values("miou", ascending=False).head(8).to_string(index=False))
    print(f"fit {fit_min:.1f} min, best lambda={best['lambda']:g} sigma={best['sigma']:g} bias={best['bias']:g} "
          f"val mIoU={best['miou']:.2f}")


if __name__ == "__main__":
    main()
