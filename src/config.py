"""Central configuration: paths and hyperparameters.

Single source of truth shared by the training scripts and (a minimal, torch-free
subset by) the serving app. Values reflect the locked decisions in
starter_plan.md / CLAUDE.md. This module imports only the stdlib so it is safe to
pull into the slim inference image.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths ---------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "isic2018"             # downloaded ISIC 2018 Task 1 (gitignored)
MODELS_DIR = ROOT_DIR / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"      # torch checkpoints (gitignored)
ONNX_MODEL_PATH = MODELS_DIR / "segformer.onnx"  # committed via git-lfs; served in prod

# --- Model ---------------------------------------------------------------
# SegFormer backbone. b1 = accuracy/size balance for CPU Cloud Run. Fall back to
# b0 for latency, step up to b2 only if accuracy demands it.
MODEL_CHECKPOINT = "nvidia/mit-b1"
NUM_LABELS = 2                                    # binary: background / lesion
ID2LABEL = {0: "background", 1: "lesion"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# Input resolution fed to the model. 512 matches SegFormer pretraining (best
# accuracy); drop to 256 if CPU inference latency is too high for the live demo.
IMAGE_SIZE = 512

# ImageNet normalization (SegformerImageProcessor defaults). Inference pre/post in
# src/app/inference.py MUST match these or predicted masks will be wrong.
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

# --- Training ------------------------------------------------------------
SEED = 42
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 6e-5
WEIGHT_DECAY = 1e-2
DICE_WEIGHT = 0.5            # loss = DICE_WEIGHT * Dice + (1 - DICE_WEIGHT) * CrossEntropy
NUM_WORKERS = 4

# --- Serving -------------------------------------------------------------
DEFAULT_PORT = 8080         # Cloud Run injects $PORT; this is the local/Docker default
