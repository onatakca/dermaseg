"""ISIC 2018 torch Dataset with albumentations augmentation.

Phase 2 — not yet implemented. Yields (image, mask) pairs with a reproducible
train/val/test split (config.SEED, config.VAL_SPLIT, config.TEST_SPLIT). Train
transforms: flips, rotations, color jitter, resize to config.IMAGE_SIZE.
"""

from __future__ import annotations
