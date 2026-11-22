"""Rasterize COCO polygon annotations (e.g. a makesense.ai export) into 0/1 PNG masks for data/local."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import DATA_DIR


def main() -> None:
    """Draw every polygon of each image into one mask at the annotated image size and save it beside the images."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("labels", type=Path, nargs="?", default=DATA_DIR / "local" / "labels.json")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "local" / "masks")
    args = ap.parse_args()
    coco = json.loads(args.labels.read_text(encoding="utf-8"))
    canvases = {}
    for image in coco["images"]:
        canvases[image["id"]] = (Path(image["file_name"]).stem, Image.new("L", (image["width"], image["height"]), 0))
    for ann in coco["annotations"]:
        draw = ImageDraw.Draw(canvases[ann["image_id"]][1])
        for poly in ann["segmentation"]:
            draw.polygon(list(zip(poly[0::2], poly[1::2])), fill=1)
    args.out.mkdir(parents=True, exist_ok=True)
    for stem, mask in canvases.values():
        mask.save(args.out / f"{stem}.png")
        print(f"{stem}: {100 * np.asarray(mask).mean():.1f}% water")


if __name__ == "__main__":
    main()
