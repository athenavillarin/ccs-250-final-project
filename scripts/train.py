"""Train the U-Net from scratch or fine-tune SegFormer-B0 on a train split; saves weights, history, and curves."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import FloodDataset
from src.eval.benchmark import count_params
from src.methods import METHODS
from src.methods.pretrained import SegFormer
from src.methods.scratch import UNet
from src.utils.paths import CHECKPOINT_DIR, RESULTS_DIR
from src.utils.runs import log_run
from src.utils.seed import SEED, seed_everything
from src.utils.train import train
from src.utils.viz import plot_curves

DEFAULTS = {
    "scratch": {"epochs": 80, "lr": 1e-3, "weight_decay": 1e-4, "patience": 12},
    "pretrained": {"epochs": 50, "lr": 6e-5, "weight_decay": 1e-2, "patience": 10},
}


def build(method: str):
    """Return a fresh model for the method (random init or pretrained encoder)."""
    return UNet() if method == "scratch" else SegFormer()


def main() -> None:
    """Parse arguments, train with early stopping, and write artifacts."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=DEFAULTS, required=True)
    ap.add_argument("--split", default="train", help="train, train_50, train_25, or mixed (FloodNet + Kaggle + roadway)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--weight-decay", type=float)
    ap.add_argument("--patience", type=int)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    for key, value in DEFAULTS[args.method].items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    seed_everything(args.seed)
    tag = f"{args.method}_{args.split}_s{args.seed}"
    model = build(args.method)
    history, minutes = train(
        model, FloodDataset(args.split, train=True), FloodDataset("mixed_val" if args.split == "mixed" else "val"), CHECKPOINT_DIR / f"{tag}.pt",
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size,
        patience=args.patience, device=args.device,
    )
    RESULTS_DIR.mkdir(exist_ok=True)
    history.to_csv(RESULTS_DIR / f"history_{tag}.csv", index=False)
    name = METHODS[args.method]["name"]
    plot_curves(history, f"{name} ({METHODS[args.method]['slot']}), {args.split}, seed {args.seed}",
                RESULTS_DIR / f"curves_{tag}.png")
    log_run(args.method, args.split, args.seed, minutes, count_params(model), history["val_miou"].max())


if __name__ == "__main__":
    main()
