# Flood Segmentation: Graph Cut vs U-Net vs SegFormer

Binary flood segmentation (flood vs non-flood) on FloodNet plus aerial and street-level flood photos, comparing a classical graph cut, a U-Net trained from scratch, and a fine-tuned SegFormer-B0 on the same splits and metrics. A FastAPI + React app runs all three on uploaded photos for flood mapping. See [results/SUMMARY.md](results/SUMMARY.md) for the numbers.

**Live demo:** https://flood-mapper-bqp3.onrender.com (free CPU instance; the first visit after 15 idle minutes takes about a minute to wake).

| Slot | Method | Module |
|---|---|---|
| A. Classical | Graph cut: Lab colour + two-scale texture + row-position GMM unaries, contrast-sensitive Potts, own Boykov-Kolmogorov max-flow | [src/methods/classical.py](src/methods/classical.py) |
| B. From scratch | U-Net in PyTorch, random init | [src/methods/scratch.py](src/methods/scratch.py) |
| C. Pretrained | SegFormer-B0 (`nvidia/mit-b0`), fine-tuned | [src/methods/pretrained.py](src/methods/pretrained.py) |

Every method exposes `load(checkpoint_path=None, device="cpu")` and `predict(model, image) -> HxW uint8 {0,1}`.

## Data

| Set | Use | Source |
|---|---|---|
| FloodNet Supervised v1.0 | UAV top-down; train / val / test (official splits: 1445 / 450 / 448) | [BinaLab/FloodNet-Supervised_v1.0](https://github.com/BinaLab/FloodNet-Supervised_v1.0) |
| Flood Area Segmentation | oblique aerial and drone; 290 images, fixed 70/15/15 split (203 / 44 / 43) | [Kaggle: faizalkarim/flood-area-segmentation](https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation) |
| Roadway Flooding | street level; 441 images, fixed 70/15/15 split (309 / 66 / 66) | [Kaggle: saurabhshahane/roadway-flooding-image-dataset](https://www.kaggle.com/datasets/saurabhshahane/roadway-flooding-image-dataset) ([Sazara et al., 2019](https://arxiv.org/abs/1909.00125)) |
| Local photos | 10 Philippine street and yard flood photos, test only, labelled by the team; used when `data/local/images/*.jpg` and `data/local/masks/*.png` exist (not published; labelling protocol in `data/local/README.md`) | Wikimedia Commons |

Local photos shown in `results/domain_examples.png` (faces and licence plates pixelated): [ph_001](https://commons.wikimedia.org/wiki/File:0781Effects_(floods)_of_Typhoon_Vamco_Calumpit,_Bulacan_47.jpg) and [ph_008](https://commons.wikimedia.org/wiki/File:0547San_Miguel,_Calumpit,_Bulacan_29.jpg), both by Judgefloro, CC0, via Wikimedia Commons.

Two training configurations use these splits: **FloodNet only** (`train`) and **mixed** (`mixed` = FloodNet train + Kaggle train + Roadway train, validated on the matching val splits). The app and the main comparison use the mixed models; the FloodNet-only models are kept for the domain-shift experiment and the data-size ablation.

FloodNet masks hold IDs 0-9 (0 = background). Flood = {1 building-flooded, 3 road-flooded, 5 water}; the mapping lives in [src/data/classes.py](src/data/classes.py). The Kaggle and Roadway masks are already binary water masks. Images are stored at 1024 px on the long side; the deep models run at 512 px and the graph cut at 256 px, and every mask is upsampled to the stored size for evaluation.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/download_data.py            # FloodNet (~13 GB download, ~0.8 GB kept) + Kaggle + Roadway
python scripts/make_splits.py              # rewrites splits/ (must match the committed files)
python -m pytest -q tests                  # max-flow vs reference solver
python scripts/sanity_classical.py         # synthetic sanity check
python scripts/fit_classical.py --split train   # fit, then tune lambda/sigma/bias on val
python scripts/fit_classical.py --split mixed
python scripts/train.py --method scratch --split mixed        # and --split train, train_50, train_25
python scripts/train.py --method pretrained --split mixed     # same four splits
python scripts/run_experiments.py          # all five experiments -> results/
```

To skip training, `python scripts/download_checkpoints.py --all` fetches all trained checkpoints. `data/` and `checkpoints/` are gitignored. Override their locations with `FLOOD_DATA_DIR` and `FLOOD_CHECKPOINT_DIR`.

### Colab

Open [notebooks/main.ipynb](notebooks/main.ipynb) in Colab with a GPU runtime and run all cells. It clones this repo, installs requirements, downloads the data to `/content/data`, and downloads the trained checkpoints from the [`checkpoints-v2` release](https://github.com/athenavillarin/ccs-250-final-project/releases/tag/checkpoints-v2) (`checkpoints-v1` holds the earlier FloodNet-only weights). Set `TRAIN = True` to retrain instead.

## Experiments

| # | Experiment | Outputs in `results/` |
|---|---|---|
| 1 | Main comparison of the mixed models on the FloodNet test split | `main_results.csv`, `per_image_test.csv` |
| 2 | Data-size ablation (25 / 50 / 100 % of FloodNet train) | `ablation_data_size.csv`, `.png` |
| 3 | Efficiency: CPU and GPU runtime, parameters, train time | `efficiency.csv`, `efficiency_runtime.png` |
| 4 | Domain shift: FloodNet-only vs mixed training on the FloodNet, Kaggle, Roadway (and local) test splits | `domain_shift.csv`, `.png`, `domain_examples.png` |
| 5 | Failure analysis: per-class flood rate, hardest cases, and mIoU on 42 test images tagged for muddy water, reflections/glare, wet roads, and shadows | `failure_per_class.csv`, `.png`, `failure_cases.png`, `hard_cases.csv`, `.png`, `hard_case_examples.png` |

The hard-condition tags are in `splits/hard_case_tags.csv`: 12 flooded FloodNet, 12 Roadway, and 8 Kaggle test images drawn at random (seed 42) plus the 10 local photos. The tags were assigned by an AI assistant from the images alone and are marked for team review in the `tagged_by` column.

Training curves are in `curves_*.png` and `history_*.csv`; graph-cut tuning and the sanity check are in `classical_tuning_*` and `sanity_classical*`.

### Training time

Wall-clock minutes on one RTX 4060 (8 GB) with a 16-core CPU, from `results/training_runs.csv`. Graph-cut times cover fitting plus the λ/σ/bias grid search on val. Deep runs stop early on val mIoU, so the epoch count is part of the cost.

| Run | Train split | Epochs (best) | Minutes | How it was timed |
|---|---|---|---|---|
| Graph cut | FloodNet only | – | 3.8 | rerun alone (the original fit overlapped GPU training) |
| Graph cut | mixed | – | 16.2 | rerun alone (same reason) |
| U-Net | FloodNet only | 53 (41) | 26.8 | original run, alone |
| U-Net | mixed | 68 (56) | 47.3 | rerun alone to the same epoch count; the deployed weights are from the original run |
| SegFormer-B0 | FloodNet only | 26 (16) | 10.9 | original run, alone |
| SegFormer-B0 | mixed | 42 (32) | 32.9 | original run; a solo rerun took 0.76 min/epoch, within 3% of this run's 0.78 |
| U-Net / SegFormer-B0 | 50 % FloodNet | 68 (56) / 50 (44) | 19.3 / 13.7 | original runs, alone |
| U-Net / SegFormer-B0 | 25 % FloodNet | 59 (47) / 50 (45) | 10.3 / 10.4 | original runs, alone |

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

The page runs the three methods on an uploaded photo (or a phone camera shot) and shows one method's mask at a time, with a before/after divider. Two extra views are computed in the browser: **Agreement** shades each pixel by how many methods call it water, and, for the five bundled sample photos (`frontend/public/samples/`, local test photos ph_003/004/006/008/010 with the team's labels), **Errors vs. labels** colours correct water, missed water, and false alarms and shows each method's mIoU on that photo.

### Deployment

The app runs on a free [Render](https://render.com) web service (512 MB RAM, 0.1 CPU) defined in [render.yaml](render.yaml) and [deploy/Dockerfile](deploy/Dockerfile). To fit that instance it skips PyTorch: `scripts/export_onnx.py` exports the U-Net and SegFormer to ONNX, and the backend runs them with ONNX Runtime when `FLOOD_RUNTIME=onnx`. On the test set the ONNX masks give the same mIoU as PyTorch. The image builds the frontend, installs only [deploy/requirements.txt](deploy/requirements.txt), downloads `classical.npz` and the ONNX models from the release, and serves the API and UI from one URL.

To deploy: sign in to Render with GitHub, choose **New > Blueprint**, select this repo, and apply. Each push to `main` redeploys. The free service sleeps after 15 minutes idle and takes about a minute to wake on the next visit.
