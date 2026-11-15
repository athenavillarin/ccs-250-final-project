"""Write the official FloodNet splits and nested train subsets for the data-size ablation."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import DATA_DIR, SPLITS_DIR
from src.utils.seed import SEED

FRACTIONS = (25, 50)


def write(name: str, ids: list[str]) -> None:
    """Save one split file with one image id per line."""
    (SPLITS_DIR / f"{name}.txt").write_text("\n".join(ids) + "\n")
    print(f"{name}: {len(ids)}")


def main() -> None:
    """List ids per official split, then draw nested 25%/50% train subsets with a fixed seed."""
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


if __name__ == "__main__":
    main()
