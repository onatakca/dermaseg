"""SegFormer build/load helpers.

Phase 3 — not yet implemented. Wraps SegformerForSemanticSegmentation with the
config.MODEL_CHECKPOINT backbone and a config.NUM_LABELS head, plus the combined
Dice + Cross-Entropy loss.
"""

from __future__ import annotations
