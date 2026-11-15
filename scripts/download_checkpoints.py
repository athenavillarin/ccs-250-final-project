"""Download the trained checkpoints from the GitHub release into CHECKPOINT_DIR."""
import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import CHECKPOINT_DIR

RELEASE_URL = "https://github.com/athenavillarin/ccs-250-final-project/releases/download/checkpoints-v1"
DEEP = ["scratch_train_s42", "pretrained_train_s42"]
ABLATION = [f"{m}_{s}_s42.pt" for m in ("scratch", "pretrained") for s in ("train_50", "train_25")]


def main() -> None:
    """Fetch each missing checkpoint, optionally the ONNX exports instead of .pt or the data-size ablation runs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--onnx", action="store_true")
    args = parser.parse_args()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    deep = [f"{name}.{'onnx' if args.onnx else 'pt'}" for name in DEEP]
    for name in ["classical.npz"] + deep + (ABLATION if args.ablation else []):
        path = CHECKPOINT_DIR / name
        if path.exists():
            continue
        print(f"downloading {name}", flush=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(f"{RELEASE_URL}/{name}", tmp)
        tmp.rename(path)


if __name__ == "__main__":
    main()
