"""Append-only log of training runs used for the train-time and ablation tables."""
import pandas as pd

from src.utils.paths import RESULTS_DIR

RUNS_CSV = RESULTS_DIR / "training_runs.csv"


def log_run(method: str, split: str, seed: int, minutes: float, params_m: float, best_val_miou: float) -> None:
    """Record one finished training run, replacing any earlier row with the same key."""
    row = pd.DataFrame([{"method": method, "split": split, "seed": seed, "train_min": round(minutes, 2),
                         "params_m": round(params_m, 3), "best_val_miou": round(best_val_miou, 2)}])
    if RUNS_CSV.exists():
        old = pd.read_csv(RUNS_CSV)
        old = old[~((old["method"] == method) & (old["split"] == split) & (old["seed"] == seed))]
        row = pd.concat([old, row], ignore_index=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    row.to_csv(RUNS_CSV, index=False)
