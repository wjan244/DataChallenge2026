"""One-time patch embedding extraction for the XGBoost DINOv3 pipeline.

Reads train_split.csv / val_split.csv / test_students.csv and serializes
patch token embeddings to disk as fp16 numpy memmaps, mirroring the pattern
in src/dino/embed.py.

Usage:
    python scripts/extract_embeddings.py --config dino_xgb.yaml
"""
import json
import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import v2
from transformers import AutoModel, AutoImageProcessor

from src.config import DEVICE, IMG_DIR, CSV_DIR, DATA
from src.dino.utils import load_config


def _build_transform(model_name: str):
    processor = AutoImageProcessor.from_pretrained(model_name)
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])


def extract_patches(backbone, df, transform, patch_start, n_patches, embed_dim,
                    patches_path, batch_size):
    patches_map = None
    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i:i + batch_size]
        images = [Image.open(IMG_DIR / fn).convert("RGB") for fn in batch_df["filename"]]
        batch = torch.stack([transform(img) for img in images]).to(DEVICE)
        with torch.no_grad():
            hs = backbone(pixel_values=batch).last_hidden_state  # (B, 1+n_reg+N, D)
        patches = hs[:, patch_start:, :].cpu().half().numpy()    # (B, N, D)
        if patches_map is None:
            patches_map = np.memmap(patches_path, dtype=np.float16, mode="w+",
                                    shape=(len(df), n_patches, embed_dim))
        end = min(i + batch_size, len(df))
        patches_map[i:end] = patches
    if patches_map is not None:
        del patches_map  # flush to disk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="dino_xgb.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    emb_dir = DATA / cfg["embedding_dir"]
    emb_dir.mkdir(parents=True, exist_ok=True)

    print("Loading backbone...")
    backbone = AutoModel.from_pretrained(cfg["model_name"]).to(DEVICE).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    n_reg = getattr(backbone.config, "num_register_tokens", 0)
    patch_start = 1 + n_reg
    embed_dim = backbone.config.hidden_size
    img_size = getattr(backbone.config, "image_size", 224)
    patch_size = backbone.config.patch_size
    n_patches = (img_size // patch_size) ** 2
    print(f"embed_dim={embed_dim}  n_patches={n_patches}  patch_start={patch_start}")

    transform = _build_transform(cfg["model_name"])
    batch_size = cfg.get("embed_batch_size", 64)

    splits: dict[str, pd.DataFrame] = {
        "train": pd.read_csv(CSV_DIR / "train_split.csv"),
        "val":   pd.read_csv(CSV_DIR / "val_split.csv"),
    }
    test_csv = CSV_DIR / "test_students.csv"
    if test_csv.exists():
        splits["test"] = pd.read_csv(test_csv)

    shapes = {}
    for split, df in splits.items():
        df = df.reset_index(drop=True)
        print(f"\nExtracting {split} ({len(df)} images)...")
        extract_patches(backbone, df, transform, patch_start, n_patches, embed_dim,
                        emb_dir / f"{split}_patches.bin", batch_size)
        df.to_csv(emb_dir / f"{split}_meta.csv", index=False)
        shapes[split] = {"N": len(df), "n_patches": n_patches, "embed_dim": embed_dim}
        print(f"  → {emb_dir}/{split}_patches.bin  shape=({len(df)}, {n_patches}, {embed_dim})")

    with open(emb_dir / "shapes.json", "w") as f:
        json.dump(shapes, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
