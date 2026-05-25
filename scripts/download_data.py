"""Download ISIC 2018 Task 1 (lesion boundary segmentation) into data/.

Stdlib only — runnable before the training deps are installed. Fetches the
official public ISIC 2018 training set and lays it out as src/data/dataset.py
expects:

    data/isic2018/
    ├── images/   ISIC_*.jpg               (dermoscopic inputs, ~11 GB)
    └── masks/    ISIC_*_segmentation.png   (binary ground truth, ~27 MB)

The downloaded zip is deleted after extraction; re-runs skip work already done.

Usage:
    python scripts/download_data.py [--force]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

# Make `src` importable when run as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

BASE_URL = "https://isic-challenge-data.s3.amazonaws.com/2018"
# (zip filename, destination subdir under config.RAW_DATA_DIR, file extension to keep)
SOURCES = [
    ("ISIC2018_Task1-2_Training_Input.zip", "images", ".jpg"),
    ("ISIC2018_Task1_Training_GroundTruth.zip", "masks", ".png"),
]
CHUNK = 1 << 20  # 1 MiB


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def download(url: str, dest: Path) -> None:
    """Stream `url` to `dest` (atomic via a .part file) with a progress line."""
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "dermaseg/0.1"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        with open(part, "wb") as fh:
            while chunk := resp.read(CHUNK):
                fh.write(chunk)
                done += len(chunk)
                pct = f"{done / total * 100:5.1f}%" if total else "  ?  "
                print(
                    f"\r  {pct}  {_human(done)} / {_human(total)}",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
    print(file=sys.stderr)
    part.replace(dest)


def extract_flat(zip_path: Path, target: Path, keep_ext: str) -> int:
    """Extract members ending in `keep_ext` straight into `target` (flattened)."""
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.endswith("/") or not member.lower().endswith(keep_ext):
                continue
            with zf.open(member) as src, open(target / Path(member).name, "wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the ISIC 2018 Task 1 dataset")
    parser.add_argument(
        "--force", action="store_true", help="re-download and re-extract even if present"
    )
    args = parser.parse_args()

    raw = config.RAW_DATA_DIR
    raw.mkdir(parents=True, exist_ok=True)

    for zip_name, subdir, ext in SOURCES:
        target = raw / subdir
        existing = list(target.glob(f"*{ext}")) if target.exists() else []
        if existing and not args.force:
            print(f"[skip] {subdir}/ already has {len(existing)} {ext} files (--force to redo)")
            continue

        zip_path = raw / zip_name
        if not zip_path.exists() or args.force:
            print(f"[download] {BASE_URL}/{zip_name}")
            download(f"{BASE_URL}/{zip_name}", zip_path)

        print(f"[extract] {zip_name} -> {subdir}/")
        n = extract_flat(zip_path, target, ext)
        zip_path.unlink()
        print(f"[done] {n} files in {target}")

    n_img = len(list((raw / "images").glob("*.jpg")))
    n_msk = len(list((raw / "masks").glob("*.png")))
    print(f"\nReady: {n_img} images, {n_msk} masks under {raw}")
    if n_img != n_msk:
        print("WARNING: image and mask counts differ — check the download.", file=sys.stderr)


if __name__ == "__main__":
    main()
