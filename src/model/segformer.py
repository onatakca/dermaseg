"""SegFormer model builder and training loss (Dice + Cross-Entropy).

build_model() loads the MiT backbone (config.MODEL_CHECKPOINT) with a freshly
initialized config.NUM_LABELS segmentation head. The loss combines soft Dice and
Cross-Entropy to handle the lesion/background class imbalance. Loss functions
expect logits already upsampled to the target mask resolution.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

from src import config


def build_model() -> SegformerForSemanticSegmentation:
    """Load the MiT backbone with a fresh 2-class segmentation head."""
    return SegformerForSemanticSegmentation.from_pretrained(
        config.MODEL_CHECKPOINT,
        num_labels=config.NUM_LABELS,
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
        ignore_mismatched_sizes=True,
    )


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft multi-class Dice loss. logits (B,C,H,W); target (B,H,W) int64."""
    num_classes = logits.shape[1]
    probs = logits.softmax(dim=1)
    target_1h = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * target_1h).sum(dims)
    cardinality = probs.sum(dims) + target_1h.sum(dims)
    dice = (2 * intersection + eps) / (cardinality + eps)
    return 1 - dice.mean()


def combined_loss(
    logits: torch.Tensor, target: torch.Tensor, dice_weight: float = config.DICE_WEIGHT
) -> torch.Tensor:
    """dice_weight * Dice + (1 - dice_weight) * CrossEntropy (logits at target res)."""
    ce = F.cross_entropy(logits, target)
    return dice_weight * dice_loss(logits, target) + (1 - dice_weight) * ce
