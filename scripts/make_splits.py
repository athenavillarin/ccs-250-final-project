"""Write the official FloodNet splits, nested train subsets for the ablation, and fixed splits for the extra sets."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import DATA_DIR, SPLITS_DIR
from src.utils.seed import SEED

FRACTIONS = (25, 50)
EXTRA_SETS = ("kaggle", "roadway")
EXTRA_PARTS = (("train", 70), ("val", 15), ("test", 15))


def write(name: str, ids: list[str]) -> None:
    """Save one split file with one image id per line."""
    (SPLITS_DIR / f"{name}.txt").write_text("\n".join(ids) + "\n")
    print(f"{name}: {len(ids)}")


def main() -> None:
    """List FloodNet ids, draw nested train subsets, and split each extra set 70/15/15 with a fixed seed."""
    SPLITS_DIR.mkdir(exist_ok=True)
    for split in ("train", "val", "test"):
        ids = sorted(p.stem for p in (DATA_DIR / "floodnet" / split / "masks").glob("*.png"))
        if not ids:
            raise FileNotFoundError(f"no masks for split '{split}'; run scripts/download_data.py first")
        write(split, ids)
    train = sorted((SPLITS_DIR / "train.txt").read_text().split())
    random.Random(SEED).shuffle(train)
    for frac in FRACTIONS:
        write(f"train_{frac}", sorted(train[: len(train) * frac // 100]))
    for name in EXTRA_SETS:
        ids = sorted((SPLITS_DIR / f"{name}.txt").read_text().split(), key=int)
        random.Random(SEED).shuffle(ids)
        start = 0
        for part, pct in EXTRA_PARTS:
            end = len(ids) if part == "test" else start + round(len(ids) * pct / 100)
            write(f"{name}_{part}", sorted(ids[start:end], key=int))
            start = end
    local = sorted(p.stem for p in (DATA_DIR / "local" / "masks").glob("*.png"))
    if local:
        write("local", local)


if __name__ == "__main__":
    main()
