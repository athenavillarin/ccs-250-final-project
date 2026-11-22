"""Run the five experiments (main, ablation, efficiency, domain shift, failure) and write tables and figures."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.classes import FLOOD_IDS, FLOODNET_CLASSES
from src.data.dataset import load_pair, read_split, split_dir
from src.eval.benchmark import count_params
from src.eval.metrics import confusion, scores
from src.methods import METHODS
from src.utils.paths import CHECKPOINT_DIR, DATA_DIR, RESULTS_DIR, SPLITS_DIR
from src.utils.runs import RUNS_CSV
from src.utils.seed import SEED
from src.utils.viz import INK_2, METHOD_COLORS, overlay, plt, save

PRED_DIR = DATA_DIR / "preds"
DEEP = ("scratch", "pretrained")
FRACTIONS = {"train_25": 25, "train_50": 50, "train": 100}
MAIN = "mixed"
TRAIN_LABELS = {"train": "FloodNet only", "mixed": "FloodNet + Kaggle + roadway"}
TEST_SETS = {"test": "FloodNet (UAV)", "kaggle_test": "Kaggle (aerial)", "roadway_test": "Roadway (street)",
             "local": "Local photos"}


def sync(device: str) -> None:
    """Wait for queued GPU work so per-image timings are accurate."""
    if device.startswith("cuda"):
        import torch
        torch.cuda.synchronize()


def checkpoint(key: str, train_split: str = MAIN) -> str:
    """Checkpoint path for a method trained on a given split."""
    if key == "classical":
        return str(CHECKPOINT_DIR / f"classical_{train_split}.npz")
    return str(CHECKPOINT_DIR / f"{key}_{train_split}_s{SEED}.pt")


def predict_split(key: str, split: str, device: str, train_split: str = MAIN) -> pd.DataFrame:
    """Predict every image of a split once, caching masks and per-image runtimes under data/preds/."""
    out = PRED_DIR / f"{key}_{train_split}" / split
    times_csv = out / "times.csv"
    ids = read_split(split)
    if times_csv.exists() and len(pd.read_csv(times_csv)) == len(ids):
        return pd.read_csv(times_csv, dtype={"id": str})
    out.mkdir(parents=True, exist_ok=True)
    module = METHODS[key]["module"]
    model = module.load(checkpoint(key, train_split), device)
    rows = []
    for image_id in tqdm(ids, desc=f"{key}/{train_split} on {split}"):
        image, _ = load_pair(split, image_id)
        sync(device)
        start = time.perf_counter()
        pred = module.predict(model, image)
        sync(device)
        rows.append({"id": image_id, "ms": 1000 * (time.perf_counter() - start)})
        Image.fromarray(pred).save(out / f"{image_id}.png")
    df = pd.DataFrame(rows)
    df.to_csv(times_csv, index=False)
    return df


def load_pred(key: str, split: str, image_id: str, train_split: str = MAIN) -> np.ndarray:
    """Read one cached prediction."""
    return np.asarray(Image.open(PRED_DIR / f"{key}_{train_split}" / split / f"{image_id}.png"))


def score_split(key: str, split: str, train_split: str = MAIN) -> tuple[dict, list[dict]]:
    """Pooled scores over a split plus per-image scores, from cached predictions."""
    total, rows = np.zeros((2, 2), np.int64), []
    for image_id in read_split(split):
        _, gt = load_pair(split, image_id)
        cm = confusion(load_pred(key, split, image_id, train_split), gt)
        total += cm
        rows.append({"method": key, "id": image_id, "gt_flood_pct": 100 * gt.mean(), **scores(cm)})
    return scores(total), rows


def main_table(device: str, split: str = "test") -> pd.DataFrame:
    """Experiment 1: the comparison table of the deployed (mixed-trained) models on one split."""
    runs = pd.read_csv(RUNS_CSV) if RUNS_CSV.exists() else pd.DataFrame(columns=["method", "split", "train_min"])
    rows, per_image = [], []
    for key, info in METHODS.items():
        dev = "cpu" if key == "classical" else device
        times = predict_split(key, split, dev)
        pooled, images = score_split(key, split)
        per_image += images
        train = runs[(runs["method"] == key) & (runs["split"] == MAIN)]
        model = info["module"].load(checkpoint(key), "cpu")
        rows.append({"method": key, "name": info["name"], "slot": info["slot"], **pooled,
                     "runtime_ms": times["ms"].iloc[1:].median(), "device": dev.split(":")[0].upper(),
                     "params_m": count_params(model),
                     "train_min": train["train_min"].iloc[0] if len(train) else np.nan})
    df = pd.DataFrame(rows).round(3)
    df.to_csv(RESULTS_DIR / "main_results.csv", index=False)
    pd.DataFrame(per_image).round(3).to_csv(RESULTS_DIR / f"per_image_{split}.csv", index=False)
    return df


def ablation(device: str) -> pd.DataFrame:
    """Experiment 2: test mIoU of the deep models trained on 25/50/100% of the train split."""
    rows = []
    for key in DEEP:
        for train_split, pct in FRACTIONS.items():
            if not Path(checkpoint(key, train_split)).exists():
                continue
            predict_split(key, "test", device, train_split)
            pooled, _ = score_split(key, "test", train_split)
            rows.append({"method": key, "name": METHODS[key]["name"], "train_pct": pct,
                         "train_images": len(read_split(train_split)), **pooled})
    df = pd.DataFrame(rows).round(3)
    df.to_csv(RESULTS_DIR / "ablation_data_size.csv", index=False)
    predict_split("classical", "test", "cpu", "train")
    classical = score_split("classical", "test", "train")[0]["miou"]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.axhline(classical, color=METHOD_COLORS["classical"], linestyle="--", linewidth=1.5,
               label=f"{METHODS['classical']['name']} (fit on full train)")
    for key in DEEP:
        g = df[df["method"] == key]
        ax.plot(g["train_images"], g["miou"], marker="o", color=METHOD_COLORS[key], label=METHODS[key]["name"])
    ax.set_xlabel("Training images (count)")
    ax.set_ylabel("Test mIoU (%)")
    ax.set_title("Data-size ablation on the FloodNet test split")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    save(fig, RESULTS_DIR / "ablation_data_size.png")
    return df


def efficiency(device: str, n_images: int = 30) -> pd.DataFrame:
    """Experiment 3: runtime per image on CPU and GPU, parameters, and training time."""
    ids = read_split("test")[:n_images]
    images = [load_pair("test", i)[0] for i in ids]
    runs = pd.read_csv(RUNS_CSV)
    rows = []
    for key, info in METHODS.items():
        for dev in ("cpu",) if key == "classical" else ("cpu", device):
            model = info["module"].load(checkpoint(key), dev)
            times = []
            for i, image in enumerate(images):
                sync(dev)
                start = time.perf_counter()
                info["module"].predict(model, image)
                sync(dev)
                if i:
                    times.append(1000 * (time.perf_counter() - start))
            train = runs[(runs["method"] == key) & (runs["split"] == MAIN)]
            rows.append({"method": key, "name": info["name"], "device": dev.split(":")[0].upper(),
                         "runtime_ms_median": np.median(times), "runtime_ms_p90": np.percentile(times, 90),
                         "params_m": count_params(model),
                         "train_min": train["train_min"].iloc[0] if len(train) else np.nan})
    df = pd.DataFrame(rows).round(3)
    df.to_csv(RESULTS_DIR / "efficiency.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    labels = [f"{r['name']} ({r['device']})" for _, r in df.iterrows()]
    ax.barh(labels, df["runtime_ms_median"], color=[METHOD_COLORS[m] for m in df["method"]], height=0.6)
    for y, v in enumerate(df["runtime_ms_median"]):
        ax.annotate(f"{v:.0f}", (v, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9)
    ax.set_xscale("log")
    ticks = [t for t in (10, 30, 100, 300, 1000, 3000) if t <= 2 * df["runtime_ms_median"].max()]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Median runtime per image (ms, log scale)")
    ax.set_title(f"Inference time on {n_images - 1} FloodNet test images")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    save(fig, RESULTS_DIR / "efficiency_runtime.png")
    return df


def test_sets() -> list[str]:
    """Held-out test splits to evaluate, including data/local when it has masks."""
    has_local = (split_dir("local") / "masks").is_dir() and (SPLITS_DIR / "local.txt").exists()
    return [t for t in TEST_SETS if t != "local" or has_local]


def domain(device: str) -> pd.DataFrame:
    """Experiment 4: FloodNet-only vs mixed training, scored on every held-out test set."""
    rows = []
    for key, info in METHODS.items():
        dev = "cpu" if key == "classical" else device
        for train_split in TRAIN_LABELS:
            for test_split in test_sets():
                predict_split(key, test_split, dev, train_split)
                pooled, _ = score_split(key, test_split, train_split)
                rows.append({"method": key, "name": info["name"], "train": train_split, "test": test_split, **pooled})
    df = pd.DataFrame(rows).round(3)
    df.to_csv(RESULTS_DIR / "domain_shift.csv", index=False)

    sets = test_sets()
    fig, axes = plt.subplots(1, len(sets), figsize=(3.6 * len(sets), 3.6), sharey=True, squeeze=False)
    x = np.arange(len(METHODS))
    for ax, test_split in zip(axes[0], sets):
        g = df[df["test"] == test_split].set_index(["method", "train"])["miou"]
        for offset, train_split, style in ((-0.2, "train", {"alpha": 0.45, "hatch": "//", "edgecolor": "white"}),
                                           (0.2, MAIN, {})):
            vals = [g[(key, train_split)] for key in METHODS]
            ax.bar(x + offset, vals, 0.38, color=[METHOD_COLORS[k] for k in METHODS], **style)
            for xi, v in zip(x, vals):
                ax.annotate(f"{v:.0f}", (xi + offset, v), xytext=(0, 2), textcoords="offset points", ha="center",
                            fontsize=8)
        ax.set_xticks(x, [info["name"] for info in METHODS.values()], fontsize=8)
        ax.set_title(TEST_SETS[test_split])
        ax.set_ylim(0, 100)
        ax.grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("Test mIoU (%)")
    handles = [Patch(facecolor=INK_2, alpha=0.45, hatch="//", edgecolor="white", label=TRAIN_LABELS["train"]),
               Patch(facecolor=INK_2, label=TRAIN_LABELS[MAIN])]
    fig.legend(handles=handles, title="Training data", loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2)
    fig.suptitle("Effect of adding aerial and street-level training data", y=1.02)
    save(fig, RESULTS_DIR / "domain_shift.png")

    picks = []
    for test_split in sets:
        ids = [i for i in read_split(test_split) if load_pair(test_split, i)[1].mean() > 0.1]
        rng = np.random.default_rng(SEED)
        picks += [(test_split, str(i)) for i in rng.choice(ids, min(2, len(ids)), replace=False)]
    domain_grid(picks, RESULTS_DIR / "domain_examples.png")
    return df


def redact(image: np.ndarray, split: str, image_id: str, block: int = 12) -> np.ndarray:
    """Pixelate boxes listed for this image in the set's redact.json (faces, plates); display only, never scored."""
    path = split_dir(split) / "redact.json"
    if not path.exists():
        return image
    out = Image.fromarray(image)
    for x0, y0, x1, y1 in json.loads(path.read_text(encoding="utf-8")).get(image_id, []):
        region = out.crop((x0, y0, x1, y1))
        small = region.resize((max(1, (x1 - x0) // block), max(1, (y1 - y0) // block)), Image.BILINEAR)
        out.paste(small.resize(region.size, Image.NEAREST), (x0, y0))
    return np.asarray(out)


def domain_grid(picks: list[tuple[str, str]], path: Path) -> None:
    """Rows of input, ground truth, and each method's mask under both training sets for the same images."""
    cols = 2 + 2 * len(METHODS)
    fig, axes = plt.subplots(len(picks), cols, figsize=(2.6 * cols, 2.3 * len(picks)), squeeze=False)
    for row, (test_split, image_id) in zip(axes, picks):
        image, gt = load_pair(test_split, image_id)
        image = redact(image, test_split, image_id)
        panels = [(f"Input, {TEST_SETS[test_split]}", image), ("Ground truth", overlay(image, gt))]
        for key, info in METHODS.items():
            for train_split in TRAIN_LABELS:
                pred = load_pred(key, test_split, image_id, train_split)
                miou = scores(confusion(pred, gt))["miou"]
                tag = "FloodNet" if train_split == "train" else "mixed"
                panels.append((f"{info['name']}, {tag} ({miou:.0f}%)", overlay(image, pred)))
        for ax, (label, img) in zip(row, panels):
            ax.imshow(img)
            ax.set_title(label, fontsize=8)
            ax.axis("off")
    fig.suptitle("Same images, FloodNet-only vs mixed training", y=1.01)
    fig.tight_layout(h_pad=2.0)
    save(fig, path)


def failure(n_cases: int = 4) -> pd.DataFrame:
    """Experiment 5: flood-prediction rate per original FloodNet class, plus hardest-case overlays."""
    counts = {key: np.zeros((len(FLOODNET_CLASSES), 2), np.int64) for key in METHODS}
    ids = read_split("test")
    for image_id in tqdm(ids, desc="per-class"):
        label = np.asarray(Image.open(split_dir("test") / "masks" / f"{image_id}.png")).ravel()
        for key in METHODS:
            pred = load_pred(key, "test", image_id).ravel()
            counts[key][:, 0] += np.bincount(label, minlength=len(FLOODNET_CLASSES))
            counts[key][:, 1] += np.bincount(label, weights=pred, minlength=len(FLOODNET_CLASSES)).astype(np.int64)
    rows = []
    for key, c in counts.items():
        for cid, name in FLOODNET_CLASSES.items():
            if c[cid, 0]:
                rows.append({"method": key, "class_id": cid, "class": name, "is_flood": cid in FLOOD_IDS,
                             "pixels": int(c[cid, 0]), "pred_flood_pct": 100 * c[cid, 1] / c[cid, 0]})
    df = pd.DataFrame(rows).round(3)
    df.to_csv(RESULTS_DIR / "failure_per_class.csv", index=False)

    order = df[df["method"] == "classical"].sort_values(["is_flood", "class_id"])
    classes = order["class"].tolist()
    y = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for i, key in enumerate(METHODS):
        vals = df[df["method"] == key].set_index("class").loc[classes, "pred_flood_pct"]
        ax.barh(y - (i - 1) * 0.27, vals, 0.25, color=METHOD_COLORS[key], label=METHODS[key]["name"])
    ax.set_yticks(y, [f"{c} (flood)" if f else c for c, f in zip(classes, order["is_flood"])])
    ax.set_xlabel("Pixels predicted as flood (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Per-class flood rate on the test split\n(high is good for flood classes, low for the rest)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, RESULTS_DIR / "failure_per_class.png")

    per_image = pd.read_csv(RESULTS_DIR / "per_image_test.csv", dtype={"id": str})
    flooded = per_image[per_image["gt_flood_pct"] > 5]
    hardest = flooded.groupby("id")["miou"].mean().nsmallest(n_cases).index
    easiest = flooded.groupby("id")["miou"].mean().nlargest(2).index
    overlay_grid(list(hardest), RESULTS_DIR / "failure_cases.png", "Hardest flooded test images (lowest mean mIoU)")
    overlay_grid(list(easiest), RESULTS_DIR / "qualitative_examples.png", "Easiest flooded test images")
    return df


def overlay_grid(ids: list[str], path: Path, title: str) -> None:
    """Rows of input, ground truth, and each method's mask overlay with per-image mIoU."""
    per_image = pd.read_csv(RESULTS_DIR / "per_image_test.csv", dtype={"id": str}).set_index(["method", "id"])
    fig, axes = plt.subplots(len(ids), 2 + len(METHODS), figsize=(3.2 * (2 + len(METHODS)), 2.6 * len(ids)), squeeze=False)
    for row, image_id in zip(axes, ids):
        image, gt = load_pair("test", image_id)
        panels = [("Input", image), ("Ground truth", overlay(image, gt))]
        for key in METHODS:
            miou = per_image.loc[(key, image_id), "miou"]
            panels.append((f"{METHODS[key]['name']} ({miou:.0f}%)", overlay(image, load_pred(key, "test", image_id))))
        for ax, (label, img) in zip(row, panels):
            ax.imshow(img)
            ax.set_title(label, fontsize=9)
            ax.axis("off")
        row[0].set_ylabel(image_id)
    fig.suptitle(title)
    fig.tight_layout(h_pad=2.5)
    save(fig, path)


def summary() -> None:
    """Write results/SUMMARY.md with the key numbers from every experiment."""
    lines = ["# Results summary", "", "Numbers only; generated by `scripts/run_experiments.py`.", ""]
    for title, name in [("1. Main comparison (FloodNet test, mixed-trained models)", "main_results.csv"),
                        ("2. Data-size ablation (FloodNet-only training, FloodNet test)", "ablation_data_size.csv"),
                        ("3. Efficiency", "efficiency.csv"),
                        ("4. Domain shift: FloodNet-only vs mixed training on each test set", "domain_shift.csv"),
                        ("5. Failure analysis: % of each class predicted as flood", "failure_per_class.csv")]:
        path = RESULTS_DIR / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if name == "failure_per_class.csv":
            df = df.pivot(index="class", columns="method", values="pred_flood_pct").reset_index()
        lines += [f"## {title}", "", df.round(2).to_markdown(index=False), ""]
    runs = RESULTS_DIR / "training_runs.csv"
    if runs.exists():
        lines += ["## Training runs", "", pd.read_csv(runs).to_markdown(index=False), ""]
    (RESULTS_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the selected experiments in dependency order."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", nargs="+", default=["main", "domain", "ablation", "efficiency", "failure"],
                    choices=["main", "domain", "ablation", "efficiency", "failure"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if "main" in args.exp:
        print(main_table(args.device).to_string(index=False))
    if "domain" in args.exp:
        print(domain(args.device).to_string(index=False))
    if "ablation" in args.exp:
        print(ablation(args.device).to_string(index=False))
    if "efficiency" in args.exp:
        print(efficiency(args.device).to_string(index=False))
    if "failure" in args.exp:
        failure()
    summary()


if __name__ == "__main__":
    main()
