"""Create a report-ready figure for the main mixed-model comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT_DIR = RESULTS / "figures"


def main() -> None:
    df = pd.read_csv(RESULTS / "main_results.csv")
    df["label"] = df["name"].replace({"Graph cut": "Graph Cut"})

    colors = ["#4C78A8", "#F58518", "#54A24B"]
    metrics = [
        ("miou", "mIoU (%)", "Mean IoU"),
        ("iou_flood", "Flood IoU (%)", "Flood IoU"),
        ("dice", "Dice", "Flood Dice"),
        ("pixel_acc", "Pixel accuracy (%)", "Pixel accuracy"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.4), constrained_layout=True)
    for ax, (column, ylabel, title) in zip(axes, metrics):
        values = df[column].tolist()
        bars = ax.bar(df["label"], values, color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(df)), df["label"], rotation=20, ha="right")
        maximum = 1.0 if column == "dice" else 100.0
        ax.set_ylim(0, maximum * 1.12)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        for bar, value in zip(bars, values):
            label = f"{value:.2f}" if column != "dice" else f"{value:.3f}"
            ax.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle("Main model comparison on the FloodNet test split", fontsize=13)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "main_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
