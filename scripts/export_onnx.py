"""Export the best checkpoint to ONNX and verify torch vs onnxruntime parity.

Phase 5 — not yet implemented. Writes config.ONNX_MODEL_PATH (committed via
git-lfs) and asserts onnxruntime outputs match the torch model within tolerance.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Phase 5: implement ONNX export + parity check")


if __name__ == "__main__":
    main()
