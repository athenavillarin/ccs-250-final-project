# Flood Segmentation: Graph Cut vs U-Net vs SegFormer

Binary flood segmentation (flood vs non-flood) on FloodNet, comparing a classical graph cut, a U-Net trained from scratch, and a fine-tuned SegFormer-B0 on the same splits and metrics. A FastAPI + React app runs all three on uploaded photos for flood mapping. See [results/SUMMARY.md](results/SUMMARY.md) for the numbers.

| Slot | Method | Module |
|---|---|---|
| A. Classical | Graph cut: Lab colour + texture GMM unaries, contrast-sensitive Potts, own Boykov-Kolmogorov max-flow | [src/methods/classical.py](src/methods/classical.py) |
| B. From scratch | U-Net in PyTorch, random init | [src/methods/scratch.py](src/methods/scratch.py) |
| C. Pretrained | SegFormer-B0 (`nvidia/mit-b0`), fine-tuned | [src/methods/pretrained.py](src/methods/pretrained.py) |

Every method exposes `load(checkpoint_path=None, device="cpu")` and `predict(model, image) -> HxW uint8 {0,1}`.

## Data

| Set | Use | Source |
|---|---|---|
| FloodNet Supervised v1.0 | train / val / test (official splits: 1445 / 450 / 448) | [BinaLab/FloodNet-Supervised_v1.0](https://github.com/BinaLab/FloodNet-Supervised_v1.0) |
| Flood Area Segmentation | out-of-domain test (290 images) | [Kaggle: faizalkarim/flood-area-segmentation](https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation) |

FloodNet masks hold IDs 0-9 (0 = background). Flood = {1 building-flooded, 3 road-flooded, 5 water}; the mapping lives in [src/data/classes.py](src/data/classes.py). Images are stored at 1024 px on the long side; the deep models run at 512 px and the graph cut at 256 px, and every mask is upsampled to the stored size for evaluation.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/download_data.py            # ~13 GB download, ~0.8 GB kept
python scripts/make_splits.py              # rewrites splits/ (must match the committed files)
python -m pytest -q tests                  # max-flow vs reference solver
python scripts/sanity_classical.py         # synthetic sanity check
python scripts/fit_classical.py            # fit on train, tune lambda/sigma/bias on val
python scripts/train.py --method scratch --split train
python scripts/train.py --method pretrained --split train
python scripts/train.py --method scratch --split train_50     # and train_25, for both methods
python scripts/run_experiments.py          # all five experiments -> results/
```

To skip training, `python scripts/download_checkpoints.py --ablation` fetches all trained checkpoints. `data/` and `checkpoints/` are gitignored. Override their locations with `FLOOD_DATA_DIR` and `FLOOD_CHECKPOINT_DIR`.

### Colab

Open [notebooks/main.ipynb](notebooks/main.ipynb) in Colab with a GPU runtime and run all cells. It clones this repo, installs requirements, downloads the data to `/content/data`, and downloads the trained checkpoints from the [`checkpoints-v1` release](https://github.com/athenavillarin/ccs-250-final-project/releases/tag/checkpoints-v1). Set `TRAIN = True` to retrain instead.

## Experiments

| # | Experiment | Outputs in `results/` |
|---|---|---|
| 1 | Main comparison on the FloodNet test split | `main_results.csv`, `per_image_test.csv` |
| 2 | Data-size ablation (25 / 50 / 100 % of train) | `ablation_data_size.csv`, `.png` |
| 3 | Efficiency: CPU and GPU runtime, parameters, train time | `efficiency.csv`, `efficiency_runtime.png` |
| 4 | Out-of-domain test on the Kaggle set | `ood_kaggle.csv`, `ood_comparison.png` |
| 5 | Failure analysis: per-class flood rate, hardest cases | `failure_per_class.csv`, `.png`, `failure_cases.png` |

Training curves are in `curves_*.png` and `history_*.csv`; graph-cut tuning and the sanity check are in `classical_tuning.*` and `sanity_classical*`.

## App

### Backend

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --port 8000
```

| Endpoint | Description |
|---|---|
| `GET /health` | status, device, loaded methods, load errors |
| `POST /segment` | multipart `image` plus optional `methods` (e.g. `classical,pretrained`); returns per-method RGBA mask PNG (base64), % flooded, runtime (ms), params (M) |
| `GET /recommendation` | best method by test mIoU from `results/main_results.csv`, a short explanation, and the results table |

Environment variables: `FLOOD_DEVICE` (`cpu` or `cuda`), `FLOOD_RUNTIME` (`torch` or `onnx`), `FLOOD_THREADS` (ONNX Runtime threads), `FLOOD_CORS_ORIGINS` (comma-separated, default `*`). To expose a local or Colab backend for a demo, run `cloudflared tunnel --url http://localhost:8000` and use the printed URL as `VITE_API_URL`.

### Frontend

```bash
cd frontend
cp .env.example .env        # set VITE_API_URL to the backend URL
npm install
npm run dev
```

If `frontend/dist` exists (built with `VITE_API_URL=` empty), the backend also serves the frontend at `/`.

### Deployment

The app runs on a free [Render](https://render.com) web service (512 MB RAM, 0.1 CPU) defined in [render.yaml](render.yaml) and [deploy/Dockerfile](deploy/Dockerfile). To fit that instance it skips PyTorch: `scripts/export_onnx.py` exports the U-Net and SegFormer to ONNX, and the backend runs them with ONNX Runtime when `FLOOD_RUNTIME=onnx`. On the test set the ONNX masks give the same mIoU as PyTorch. The image builds the frontend, installs only [deploy/requirements.txt](deploy/requirements.txt), downloads `classical.npz` and the ONNX models from the release, and serves the API and UI from one URL.

To deploy: sign in to Render with GitHub, choose **New > Blueprint**, select this repo, and apply. Each push to `main` redeploys. The free service sleeps after 15 minutes idle and takes about a minute to wake on the next visit.
