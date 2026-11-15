"""Download FloodNet Supervised v1.0 and the Kaggle Flood Area set, storing resized image/mask pairs."""
import argparse
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import DATA_DIR, SPLITS_DIR

FLOODNET_URL = "https://www.dropbox.com/scl/fo/k33qdif15ns2qv2jdxvhx/ANGaa8iPRhvlrvcKXjnmNRc?rlkey=ao2493wzl1cltonowjdbrnp7f&dl=1"
KAGGLE_DATASET = "faizalkarim/flood-area-segmentation"
SPLIT_RE =re.compile(r"(?:^|/)(train|val|test)[^/]*/", re.IGNORECASE)


def download(url: str, dest: Path) -> None:
    """Stream a URL to disk with a progress bar, skipping if the file already exists."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url) as res, tmp.open("wb") as f:
        bar = tqdm(total=int(res.headers.get("Content-Length", 0)), unit="B", unit_scale=True, desc=dest.name)
        while chunk := res.read(1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
    tmp.rename(dest)


def pairs(names: list[str]) -> list[tuple[str, str, str, str]]:
    """Match each *_lab.png label to its image, returning (split, id, image, label)."""
    images = {Path(n).stem: n for n in names if n.lower().endswith(".jpg")}
    out = []
    for name in names:
        if not name.endswith("_lab.png"):
            continue
        sid = Path(name).stem[: -len("_lab")]
        split = SPLIT_RE.search(name)
        if sid in images and split:
            out.append((split.group(1).lower(), sid, images[sid], name))
    return out


def extract(zip_path: Path, out_dir: Path, max_side: int) -> None:
    """Resize every labeled pair (bilinear image, nearest label) into out_dir/<split>/."""
    with zipfile.ZipFile(zip_path) as z:
        items = pairs(z.namelist())
        if not items:
            raise RuntimeError("no image/label pairs found; check the archive layout")
        for split, sid, img_name, lab_name in tqdm(items, desc="resize"):
            img_path = out_dir / split / "images" / f"{sid}.jpg"
            lab_path = out_dir / split / "masks" / f"{sid}.png"
            if img_path.exists() and lab_path.exists():
                continue
            img_path.parent.mkdir(parents=True, exist_ok=True)
            lab_path.parent.mkdir(parents=True, exist_ok=True)
            with z.open(img_name) as f:
                img = Image.open(f).convert("RGB")
                img.thumbnail((max_side, max_side), Image.BILINEAR)
                img.save(img_path, quality=95)
            with z.open(lab_name) as f:
                lab = Image.open(f)
                lab.thumbnail((max_side, max_side), Image.NEAREST)
                lab.save(lab_path)


def kaggle(out_dir: Path, max_side: int) -> None:
    """Fetch the Kaggle Flood Area Segmentation set and store resized images with 0/1 masks."""
    import kagglehub

    root = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    ids = []
    for mask_path in tqdm(sorted((root / "Mask").glob("*.png")), desc="kaggle"):
        sid = mask_path.stem
        img = Image.open(root / "Image" / f"{sid}.jpg").convert("RGB")
        mask = Image.open(mask_path).convert("L").point(lambda v: 1 if v >= 128 else 0)
        img.thumbnail((max_side, max_side), Image.BILINEAR)
        mask = mask.resize(img.size, Image.NEAREST)
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)
        img.save(out_dir / "images" / f"{sid}.jpg", quality=95)
        mask.save(out_dir / "masks" / f"{sid}.png")
        ids.append(sid)
    (SPLITS_DIR / "kaggle.txt").write_text("\n".join(sorted(ids, key=int)) + "\n")


def main() -> None:
    """Fetch FloodNet (and the out-of-domain Kaggle set), extract resized pairs, and drop the archive."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--keep-zip", action="store_true")
    ap.add_argument("--skip-floodnet", action="store_true")
    args = ap.parse_args()
    if not args.skip_floodnet:
        zip_path = DATA_DIR / "raw" / "FloodNet.zip"
        download(FLOODNET_URL, zip_path)
        extract(zip_path, DATA_DIR / "floodnet", args.max_side)
        if not args.keep_zip:
            zip_path.unlink()
    kaggle(DATA_DIR / "kaggle", args.max_side)


if __name__ == "__main__":
    main()
