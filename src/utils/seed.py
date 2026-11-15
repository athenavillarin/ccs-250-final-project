"""Global seeding for reproducible runs."""
import os
import random

import numpy as np

SEED = 42


def seed_everything(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch (if installed) RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
