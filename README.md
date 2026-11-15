# Flood Segmentation: Graph Cut vs U-Net vs SegFormer

Binary flood segmentation (flood vs non-flood) on FloodNet, comparing a classical graph cut, a U-Net trained from scratch, and a fine-tuned SegFormer-B0 on the same splits and metrics. A FastAPI + React app runs all three on uploaded photos for flood mapping.

## Layout

| Path | Contents |
|---|---|
| `src/methods/` | `classical.py`, `scratch.py`, `pretrained.py`, each exposing `load()` and `predict()` |
| `src/data/` | dataset loading and the binary class mapping (`classes.py`) |
| `src/eval/` | metrics (mIoU, Dice, pixel accuracy) and efficiency benchmarks |
| `splits/` | fixed train/val/test lists shared by all methods |
| `results/` | result tables (CSV) and plots (PNG) |
| `backend/` | FastAPI service |
| `frontend/` | React + Vite + TypeScript + Tailwind client |

## Setup (local)

```bash
pip install -r requirements.txt
```

Data and checkpoints live in `data/` and `checkpoints/` (gitignored). Override them with `FLOOD_DATA_DIR` and `FLOOD_CHECKPOINT_DIR`, e.g. to point at Google Drive in Colab.

## Setup (Colab)

```python
!git clone https://github.com/athenavillarin/ccs-250-final-project.git
%cd ccs-250-final-project
!pip install -q -r requirements.txt
import os
os.environ["FLOOD_DATA_DIR"] = "/content/drive/MyDrive/flood/data"
os.environ["FLOOD_CHECKPOINT_DIR"] = "/content/drive/MyDrive/flood/checkpoints"
```

## Backend

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

| Endpoint | Description |
|---|---|
| `GET /health` | status, device, loaded methods |
| `POST /segment` | multipart `image` plus optional `methods` (e.g. `classical,pretrained`); returns per-method RGBA mask PNG (base64), % flooded, runtime (ms), params (M) |
| `GET /recommendation` | best method from `results/main_results.csv` with a short explanation |

Environment variables: `FLOOD_DEVICE` (`cpu` or `cuda`), `FLOOD_CORS_ORIGINS` (comma-separated, default `*`).

## Frontend

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

`VITE_API_URL` points at the backend.
