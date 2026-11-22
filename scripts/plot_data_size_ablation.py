"""Create a report-ready figure for the training-data-size ablation."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT_DIR = RESULTS / "figures"


def main() -> None:
    df = pd.read_csv(RESULTS / "ablation_data_size.csv")
    df = df.sort_values(["method", "train_pct"])

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    styles = {
        "scratch": ("U-Net", "#F58518", "o"),
        "pretrained": ("SegFormer-B0", "#54A24B", "s"),
    }

    for method, (label, color, marker) in styles.items():
        group = df[df["method"] == method]
        ax.plot(
            group["train_pct"],
            group["miou"],
            marker=marker,
            markersize=7,
            linewidth=2,
            color=color,
            label=label,
        )
        for pct, miou in zip(group["train_pct"], group["miou"]):
            ax.annotate(
                f"{miou:.2f}",
                (pct, miou),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks([25, 50, 100], ["25%", "50%", "100%"])
    ax.set_xlabel("FloodNet training data")
    ax.set_ylabel("Test mIoU (%), axis does not start at 0")
    ax.set_title("Effect of training-data size on the FloodNet test split")
    # Zoomed axis (not from 0) so the small gaps stay readable; the y-label says so.
    low = 5 * ((df["miou"].min() - 3) // 5)
    high = 5 * -(-(df["miou"].max() + 3) // 5)
    ax.set_ylim(low, high)
    ax.grid(axis="both", linestyle=":", alpha=0.35)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "data_size_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
