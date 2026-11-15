"""Registry of the three methods, each exposing load() and predict(); modules are imported on first use of METHODS."""
from importlib import import_module

INFO = {
    "classical": {"name": "Graph cut", "slot": "A. Classical"},
    "scratch": {"name": "U-Net", "slot": "B. From scratch"},
    "pretrained": {"name": "SegFormer-B0", "slot": "C. Pretrained"},
}


def __getattr__(name: str):
    """Build METHODS lazily so torch-free code can import the classical method alone."""
    if name != "METHODS":
        raise AttributeError(name)
    return {key: {"module": import_module(f"{__name__}.{key}"), **info} for key, info in INFO.items()}
