"""Segmentation metrics on hard predictions, per image.

Each function takes class-index tensors `pred` and `target` of shape (..., H, W)
and returns a per-image tensor (shape (...,)). Foreground metrics default to the
lesion class (id 1). Shared by training validation and scripts/evaluate.py;
callers concatenate the per-image values and take the mean over the dataset.
"""

from __future__ import annotations

import torch


def dice_per_image(
    pred: torch.Tensor, target: torch.Tensor, cls: int = 1, eps: float = 1e-6
) -> torch.Tensor:
    p, t = (pred == cls), (target == cls)
    intersection = (p & t).sum(dim=(-2, -1)).float()
    cardinality = p.sum(dim=(-2, -1)).float() + t.sum(dim=(-2, -1)).float()
    return (2 * intersection + eps) / (cardinality + eps)


def iou_per_image(
    pred: torch.Tensor, target: torch.Tensor, cls: int = 1, eps: float = 1e-6
) -> torch.Tensor:
    p, t = (pred == cls), (target == cls)
    intersection = (p & t).sum(dim=(-2, -1)).float()
    union = (p | t).sum(dim=(-2, -1)).float()
    return (intersection + eps) / (union + eps)


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred == target).float().mean(dim=(-2, -1))
