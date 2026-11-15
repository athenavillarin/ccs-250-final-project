"""FastAPI service that runs all three flood segmentation methods on an uploaded image."""
import base64
import csv
import io
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.benchmark import count_params
from src.methods import METHODS
from src.utils.image import resize_long
from src.utils.paths import RESULTS_DIR

DEVICE = os.environ.get("FLOOD_DEVICE", "cpu")
MAX_UPLOAD_MB = 10
MAX_SIDE = 1024
FLOOD_RGBA = (0, 120, 255, 255)
MODELS: dict[str, object] = {}
ERRORS: dict[str, str] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and warm up every method once at startup, recording any load failures."""
    for key, info in METHODS.items():
        try:
            MODELS[key] = info["module"].load(None, DEVICE)
            info["module"].predict(MODELS[key], np.zeros((64, 64, 3), np.uint8))
        except FileNotFoundError as exc:
            ERRORS[key] = f"checkpoint not found: {Path(exc.filename).name}"
        except Exception as exc:
            ERRORS[key] = f"{type(exc).__name__}: {exc}"
    yield


app = FastAPI(title="Flood Segmentation API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("FLOOD_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_image(data: bytes) -> np.ndarray:
    """Decode an upload to RGB and shrink it so the long side is at most MAX_SIDE."""
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image.")
    return resize_long(np.asarray(img), MAX_SIDE)


def mask_png(mask: np.ndarray) -> str:
    """Encode a binary mask as a base64 RGBA PNG, colored where flooded and transparent elsewhere."""
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask.astype(bool)] = FLOOD_RGBA
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def load_results() -> list[dict]:
    """Read the main results table written by the experiment scripts, if present."""
    path = RESULTS_DIR / "main_results.csv"
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


@app.get("/health")
def health() -> dict:
    """Report service status and which methods loaded."""
    return {"status": "ok", "device": DEVICE, "loaded": sorted(MODELS), "errors": ERRORS}


@app.post("/segment")
async def segment(image: UploadFile = File(...), methods: str | None = Form(None)) -> dict:
    """Run the requested methods (comma-separated keys, default all) on one image."""
    data = await image.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Image exceeds {MAX_UPLOAD_MB} MB.")
    keys = [k.strip() for k in methods.split(",")] if methods else list(METHODS)
    unknown = [k for k in keys if k not in METHODS]
    if unknown:
        raise HTTPException(400, f"Unknown methods: {unknown}")
    rgb = read_image(data)
    results = []
    for key in keys:
        if key not in MODELS:
            results.append({"method": key, "name": METHODS[key]["name"], "error": ERRORS.get(key, "not loaded")})
            continue
        start = time.perf_counter()
        mask = METHODS[key]["module"].predict(MODELS[key], rgb)
        runtime = 1000 * (time.perf_counter() - start)
        results.append({
            "method": key,
            "name": METHODS[key]["name"],
            "mask_png": mask_png(mask),
            "flooded_pct": round(100 * float(mask.mean()), 2),
            "runtime_ms": round(runtime, 1),
            "params_m": round(count_params(MODELS[key]), 2),
        })
    return {"width": rgb.shape[1], "height": rgb.shape[0], "device": DEVICE, "results": results}


@app.get("/recommendation")
def recommendation() -> dict:
    """Name the best method by test mIoU from results/ and explain the trade-off."""
    rows = load_results()
    if not rows:
        return {"method": None, "explanation": "Results are not available yet.", "table": []}
    ranked = sorted(rows, key=lambda r: float(r["miou"]), reverse=True)
    best, runner_up = ranked[0], ranked[1]
    fastest = min(rows, key=lambda r: float(r["runtime_ms"]))
    text = (
        f"{best['name']} has the highest test mIoU ({float(best['miou']):.1f}%), "
        f"{float(best['miou']) - float(runner_up['miou']):.1f} points above {runner_up['name']}, "
        f"at {float(best['runtime_ms']):.0f} ms per image ({best['device']}) with {float(best['params_m']):.1f} M parameters."
    )
    if float(fastest["runtime_ms"]) < 0.5 * float(best["runtime_ms"]):
        text += (f" {fastest['name']} is over twice as fast ({float(fastest['runtime_ms']):.0f} ms) "
                 "if speed matters more than accuracy.")
    return {"method": best["method"], "explanation": text, "table": rows}
