"""
One-time conversion: WEBP → JPEG, preserving directory structure.
Run from the project root:
    python scripts/convert_to_jpeg.py

Output: data/Crop_224_5fp_100K_jpg/  (same tree, .jpg extension)
Then update IMG_DIR in src/config.py to point to the new folder.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm

SRC = Path("data/Crop_224_5fp_100K")
DST = Path("data/Crop_224_5fp_100K_jpg")
QUALITY = 95
NUM_WORKERS = min(8, os.cpu_count())


def convert_one(src_path: Path) -> str | None:
    rel = src_path.relative_to(SRC)
    dst_path = DST / rel.with_suffix(".jpg")
    if dst_path.exists():
        return None
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        Image.open(src_path).convert("RGB").save(dst_path, "JPEG", quality=QUALITY)
    except Exception as e:
        return f"FAILED {src_path}: {e}"
    return None


if __name__ == "__main__":
    all_files = list(SRC.rglob("*.webp"))
    print(f"Converting {len(all_files)} images with {NUM_WORKERS} workers...")

    errors = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(convert_one, p): p for p in all_files}
        for f in tqdm(as_completed(futures), total=len(futures)):
            result = f.result()
            if result:
                errors.append(result)

    print(f"Done. {len(errors)} errors.")
    for e in errors:
        print(e)