"""Registry of the three methods, each exposing load() and predict()."""
from . import classical, pretrained, scratch

METHODS = {
    "classical": {"module": classical, "name": "Graph cut", "slot": "A. Classical"},
    "scratch": {"module": scratch, "name": "U-Net", "slot": "B. From scratch"},
    "pretrained": {"module": pretrained, "name": "SegFormer-B0", "slot": "C. Pretrained"},
}
