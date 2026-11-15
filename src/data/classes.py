"""FloodNet class IDs and their mapping to the binary flood mask."""
import numpy as np

FLOODNET_CLASSES = {
    0: "background",
    1: "building-flooded",
    2: "building-non-flooded",
    3: "road-flooded",
    4: "road-non-flooded",
    5: "water",
    6: "tree",
    7: "vehicle",
    8: "pool",
    9: "grass",
}
FLOOD_IDS = (1, 3, 5)


def to_binary(label: np.ndarray) -> np.ndarray:
    """Map a FloodNet label map to a uint8 mask with flood=1, non-flood=0."""
    return np.isin(label, FLOOD_IDS).astype(np.uint8)
