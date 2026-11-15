"""Project paths, overridable with environment variables for Colab or Drive."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("FLOOD_DATA_DIR", ROOT / "data"))
CHECKPOINT_DIR = Path(os.environ.get("FLOOD_CHECKPOINT_DIR", ROOT / "checkpoints"))
SPLITS_DIR = ROOT / "splits"
RESULTS_DIR = ROOT / "results"
