"""Create a report-ready figure comparing FloodNet-only and mixed training."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT_DIR = RESULTS / "figures"

DOMAIN_LABELS = {
    "test": "FloodNet (UAV)",
    "kaggle_test": "Kaggle (aerial)",
    "roadway_test": "Roadway (street)",
    "local": "Local photos",
}
METHOD_LABELS = {
    "classical": "Graph Cut",
    "scratch": "U-Net",
    "pretrained": "SegFormer-B0",
}
METHOD_COLORS = {
    "classical": "#4C78A8",
    "scratch": "#F58518",
    "pretrained": "#54A24B",
}


def main() -> None:
    df = pd.read_csv(RESULTS / "domain_shift.csv")

    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.8), sharey=True, constrained_layout=True)
    for ax, test_name in zip(axes, DOMAIN_LABELS):
        domain = df[df["test"] == test_name]
        methods = list(METHOD_LABELS)
        x = range(len(methods))
        train_values = [
            domain[(domain["method"] == method) & (domain["train"] == "train")]["miou"].iloc[0]
            for method in methods
        ]
        mixed_values = [
            domain[(domain["method"] == method) & (domain["train"] == "mixed")]["miou"].iloc[0]
            for method in methods
        ]

        ax.bar(
            [position - 0.18 for position in x],
            train_values,
            width=0.34,
            color=[METHOD_COLORS[method] for method in methods],
            alpha=0.42,
            hatch="//",
            edgecolor="white",
            label="FloodNet only",
        )
        bars = ax.bar(
            [position + 0.18 for position in x],
            mixed_values,
            width=0.34,
            color=[METHOD_COLORS[method] for method in methods],
            label="FloodNet + Kaggle + roadway",
        )

        for bar, train_value, mixed_value in zip(bars, train_values, mixed_values):
            gain = mixed_value - train_value
            ax.annotate(
                f"{gain:+.1f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_title(DOMAIN_LABELS[test_name])
        ax.set_xticks(list(x), [METHOD_LABELS[method] for method in methods], rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle=":", alpha=0.35)

    axes[0].set_ylabel("Test mIoU (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Training data", loc="outside lower center", ncol=2)
    fig.suptitle("Effect of mixed-domain training on test mIoU", fontsize=13)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "domain_shift_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
