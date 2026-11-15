"""Export the fine-tuned U-Net and SegFormer checkpoints to ONNX for torch-free CPU deployment."""
import sys
from pathlib import Path

import onnx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import TRAIN_SIDE
from src.eval.benchmark import count_params
from src.methods import METHODS
from src.utils.paths import CHECKPOINT_DIR


def main() -> None:
    """Write checkpoints/{method}_mixed_s42.onnx with dynamic height and width plus side and params metadata."""
    for key in ("scratch", "pretrained"):
        model = METHODS[key]["module"].load(None, "cpu")
        path = CHECKPOINT_DIR / f"{key}_mixed_s42.onnx"
        dummy = torch.rand(1, 3, TRAIN_SIDE, TRAIN_SIDE)
        torch.onnx.export(model, dummy, path, input_names=["image"], output_names=["logits"], opset_version=17,
                          dynamic_axes={"image": {2: "height", 3: "width"}, "logits": {2: "height", 3: "width"}},
                          dynamo=False)
        proto = onnx.load(path)
        onnx.helper.set_model_props(proto, {"side": str(TRAIN_SIDE), "params_m": f"{count_params(model):.4f}"})
        onnx.save(proto, path)
        print(f"wrote {path.name} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
