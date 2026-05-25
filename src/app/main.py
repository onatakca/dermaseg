"""FastAPI application: serves the UI and the segmentation endpoint.

Phase 6 — not yet implemented. Routes:
  GET  /         -> static UI (src/app/static/)
  GET  /health   -> 200 liveness check
  POST /predict  -> image upload -> mask-overlay PNG + lesion-area %

Serving path runs onnxruntime only — do NOT import torch/transformers here.
"""

from __future__ import annotations
