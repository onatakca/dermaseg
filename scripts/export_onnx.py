"""Export a trained SegFormer checkpoint to ONNX and verify torch parity.

Writes config.ONNX_MODEL_PATH (the artifact served in production) and asserts the
onnxruntime output matches the torch model within tolerance. Runs on CPU so the
parity check matches the serving runtime. The exported graph outputs raw logits
at input/4 resolution; src/app/inference.py does softmax/argmax/upsample.

Example:
    python scripts/export_onnx.py --checkpoint models/checkpoints
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import SegformerForSemanticSegmentation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402


class LogitsOnly(torch.nn.Module):
    """Wrap the HF model so ONNX sees a single tensor output named `logits`."""

    def __init__(self, model: SegformerForSemanticSegmentation) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).logits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export SegFormer to ONNX with a parity check")
    p.add_argument("--checkpoint", type=Path, default=config.CHECKPOINT_DIR)
    p.add_argument("--output", type=Path, default=config.ONNX_MODEL_PATH)
    p.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--rtol", type=float, default=1e-3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    base = SegformerForSemanticSegmentation.from_pretrained(str(args.checkpoint))
    model = LogitsOnly(base).eval()
    dummy = torch.randn(1, 3, args.image_size, args.image_size)

    print(f"Exporting {args.checkpoint} -> {args.output} (opset {args.opset})")
    torch.onnx.export(
        model,
        (dummy,),
        str(args.output),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )

    with torch.no_grad():
        torch_logits = model(dummy).numpy()
    sess = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    onnx_logits = sess.run(["logits"], {"pixel_values": dummy.numpy()})[0]

    max_diff = float(np.abs(torch_logits - onnx_logits).max())
    ok = np.allclose(torch_logits, onnx_logits, atol=args.atol, rtol=args.rtol)
    print(f"Parity: max abs diff {max_diff:.2e} | within tol {ok}")
    if not ok:
        raise SystemExit(f"ONNX parity FAILED (max diff {max_diff:.2e} > atol {args.atol})")

    size_mb = args.output.stat().st_size / 1e6
    print(f"OK — wrote {args.output} ({size_mb:.1f} MB), output shape {tuple(onnx_logits.shape)}")


if __name__ == "__main__":
    main()
