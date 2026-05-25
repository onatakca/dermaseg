"""ONNX inference: load model, pre/postprocess, build the mask overlay.

Phase 6 — not yet implemented. Uses onnxruntime + numpy + pillow only.
Preprocessing (resize to config.IMAGE_SIZE, normalize with config.IMAGE_MEAN /
IMAGE_STD) MUST match scripts/train.py exactly or predicted masks will be wrong.
"""

from __future__ import annotations
