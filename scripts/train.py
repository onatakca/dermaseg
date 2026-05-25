"""Fine-tune SegFormer on ISIC 2018 with a combined Dice + Cross-Entropy loss.

Built for a CUDA GPU (Colab / cloud) with MPS/CPU fallback. Logs to Weights &
Biases (disable with --no-wandb) and saves the best checkpoint (by validation
DICE) to config.CHECKPOINT_DIR in HF format, ready for scripts/export_onnx.py.

Example (Colab, GPU runtime):
    pip install -r requirements-train.txt
    python scripts/download_data.py
    python scripts/train.py --epochs 50 --batch-size 8
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.data.dataset import get_datasets  # noqa: E402
from src.metrics import dice_per_image, iou_per_image  # noqa: E402
from src.model.segformer import build_model, combined_loss  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SegFormer on ISIC 2018 Task 1")
    p.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--output-dir", type=Path, default=config.CHECKPOINT_DIR)
    p.add_argument("--no-wandb", action="store_true", help="disable Weights & Biases logging")
    p.add_argument("--wandb-project", default="dermaseg")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def upsample(logits: torch.Tensor, size: torch.Size) -> torch.Tensor:
    """SegFormer emits logits at input/4; bring them back to the mask resolution."""
    return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses, dices, ious = [], [], []
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = upsample(model(pixel_values=images).logits, masks.shape[-2:])
        losses.append(combined_loss(logits, masks).item())
        preds = logits.argmax(dim=1)
        dices.append(dice_per_image(preds, masks).cpu())
        ious.append(iou_per_image(preds, masks).cpu())
    return {
        "loss": float(np.mean(losses)),
        "dice": float(torch.cat(dices).mean()),
        "iou": float(torch.cat(ious).mean()),
    }


def log_predictions(model, loader: DataLoader, device: torch.device, n: int = 4) -> list:
    """Build up to `n` W&B image overlays (prediction vs ground truth) from one batch."""
    import wandb

    model.eval()
    images, masks = next(iter(loader))
    with torch.no_grad():
        logits = upsample(model(pixel_values=images[:n].to(device)).logits, masks.shape[-2:])
    preds = logits.argmax(dim=1).cpu().numpy()
    mean = torch.tensor(config.IMAGE_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMAGE_STD).view(3, 1, 1)
    samples = []
    for i in range(min(n, images.shape[0])):
        img = (images[i] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        samples.append(
            wandb.Image(
                img,
                masks={
                    "prediction": {"mask_data": preds[i], "class_labels": config.ID2LABEL},
                    "ground_truth": {"mask_data": masks[i].numpy(), "class_labels": config.ID2LABEL},
                },
            )
        )
    return samples


def main() -> None:
    args = parse_args()
    set_seed(config.SEED)
    device = pick_device()
    print(f"Device: {device}")

    datasets = get_datasets()
    print("Split sizes:", {k: len(v) for k, v in datasets.items()})
    loaders = {
        split: DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=(split == "train"),
        )
        for split, ds in datasets.items()
    }

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "seed": config.SEED,
                "model": config.MODEL_CHECKPOINT,
                "image_size": config.IMAGE_SIZE,
            },
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for images, masks in loaders["train"]:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = upsample(model(pixel_values=images).logits, masks.shape[-2:])
                loss = combined_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
        scheduler.step()

        train_loss = float(np.mean(train_losses))
        val = evaluate(model, loaders["val"], device)
        print(
            f"epoch {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val['loss']:.4f} "
            f"| val_dice {val['dice']:.4f} | val_iou {val['iou']:.4f}"
        )

        if use_wandb:
            metrics = {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val["loss"],
                "val/dice": val["dice"],
                "val/iou": val["iou"],
                "lr": scheduler.get_last_lr()[0],
            }
            try:
                metrics["val/predictions"] = log_predictions(model, loaders["val"], device)
            except Exception as exc:  # never let visualization kill a long run
                print(f"  (skipped prediction logging: {exc})")
            wandb.log(metrics)

        if val["dice"] > best_dice:
            best_dice = val["dice"]
            model.save_pretrained(args.output_dir)
            print(f"  ↳ new best (val_dice {best_dice:.4f}) saved to {args.output_dir}")

    print(f"Best val DICE: {best_dice:.4f}")
    if use_wandb:
        wandb.summary["best_val_dice"] = best_dice
        wandb.finish()


if __name__ == "__main__":
    main()
