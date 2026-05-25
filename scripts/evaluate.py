"""Evaluate a trained SegFormer checkpoint on the held-out test split.

Reports lesion DICE, mean IoU (2-class), and pixel accuracy on the reproducible
test split, prints a markdown table, writes metrics.json next to the checkpoint,
and (with --readme) fills the metrics block in README.md.

Example:
    python scripts/evaluate.py --checkpoint models/checkpoints --readme
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.data.dataset import ISICDataset, list_pairs, split_pairs  # noqa: E402
from src.metrics import dice_per_image, iou_per_image, pixel_accuracy  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SegFormer on the ISIC test split")
    p.add_argument("--checkpoint", type=Path, default=config.CHECKPOINT_DIR)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--metrics-json", type=Path, default=None)
    p.add_argument("--readme", action="store_true", help="update the metrics table in README.md")
    return p.parse_args()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def run_eval(model, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    dices, mious, accs = [], [], []
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(pixel_values=images).logits
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        preds = logits.argmax(dim=1)
        dices.append(dice_per_image(preds, masks, cls=1).cpu())
        miou = (iou_per_image(preds, masks, cls=0) + iou_per_image(preds, masks, cls=1)) / 2
        mious.append(miou.cpu())
        accs.append(pixel_accuracy(preds, masks).cpu())
    return {
        "dice": float(torch.cat(dices).mean()),
        "miou": float(torch.cat(mious).mean()),
        "pixel_acc": float(torch.cat(accs).mean()),
    }


def metrics_table(m: dict[str, float]) -> str:
    return (
        "| Metric | Score |\n"
        "| --- | --- |\n"
        f"| DICE (lesion) | {m['dice']:.4f} |\n"
        f"| Mean IoU | {m['miou']:.4f} |\n"
        f"| Pixel accuracy | {m['pixel_acc']:.4f} |\n"
    )


def update_readme(table: str) -> None:
    readme = config.ROOT_DIR / "README.md"
    block = f"<!-- METRICS:START -->\n{table}<!-- METRICS:END -->"
    new, n = re.subn(
        r"<!-- METRICS:START -->.*?<!-- METRICS:END -->",
        block,
        readme.read_text(),
        count=1,
        flags=re.DOTALL,
    )
    if n == 0:
        print("WARNING: METRICS markers not found in README.md; skipped.", file=sys.stderr)
        return
    readme.write_text(new)
    print(f"Updated metrics table in {readme}")


def main() -> None:
    args = parse_args()
    device = pick_device()
    print(f"Device: {device} | checkpoint: {args.checkpoint}")

    pairs = split_pairs(list_pairs())[args.split]
    loader = DataLoader(
        ISICDataset(pairs, train=False),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    model = SegformerForSemanticSegmentation.from_pretrained(str(args.checkpoint)).to(device)

    metrics = run_eval(model, loader, device)
    print(f"\n{args.split} split ({len(pairs)} images):")
    print(metrics_table(metrics))

    json_path = args.metrics_json or (args.checkpoint / "metrics.json")
    json_path.write_text(json.dumps({"split": args.split, "n": len(pairs), **metrics}, indent=2))
    print(f"Wrote {json_path}")

    if args.readme:
        update_readme(metrics_table(metrics))


if __name__ == "__main__":
    main()
