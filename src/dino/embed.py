import torch
import pandas as pd
import argparse

from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoImageProcessor

from src.config import DEVICE, IMG_DIR, CSV_DIR, DATA
from src.dino.utils import load_config


def extract_embeddings(model, processor, df, n_reg, batch_size):
    """Returns cls [N, D] and patch_mean [N, D] tensors."""
    cls_list, patch_list = [], []
    patch_start = 1 + n_reg

    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i : i + batch_size]
        images = [Image.open(IMG_DIR / fn).convert("RGB")
          for fn in batch_df["filename"].tolist()]
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**inputs)
        hs = out.last_hidden_state                          # [B, T, D]
        cls_list.append(hs[:, 0, :].cpu())
        patch_list.append(hs[:, patch_start:, :].mean(dim=1).cpu())

    return torch.cat(cls_list), torch.cat(patch_list)


def save_split(cls, patch, df, emb_dir, split, fp16):
    dtype = torch.float16 if fp16 else torch.float32
    torch.save(cls.to(dtype),   emb_dir / f"{split}_cls.pt")
    torch.save(patch.to(dtype), emb_dir / f"{split}_patch_mean.pt")
    df.to_csv(emb_dir / f"{split}_meta.csv", index=False)
    print(f"  {split}: {cls.shape}  dtype={cls.to(dtype).dtype}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="dino.yaml",
                        help="YAML filename inside config/models/")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    emb_dir = DATA / cfg["embedding_dir"]
    emb_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    processor = AutoImageProcessor.from_pretrained(cfg["model_name"])
    model = AutoModel.from_pretrained(cfg["model_name"]).to(DEVICE).eval()
    n_reg = model.config.num_register_tokens

    # --- train / val split from train_clean.csv ---
    df_train_all = pd.read_csv(CSV_DIR / "train_clean.csv")
    df_train, df_val = train_test_split(df_train_all, test_size=0.2,
                                        random_state=42, shuffle=True)
    df_train = df_train.reset_index(drop=True)
    
    if cfg.get("include_noisy_in_val", False):
        df_noisy = pd.read_csv(CSV_DIR / "validation_noisy.csv")
        df_val = pd.concat([df_val, df_noisy], ignore_index=True)
    else:
        df_val = df_val.reset_index(drop=True)

    # --- test ---
    df_test = pd.read_csv(CSV_DIR / "test_students.csv")

    batch_size = cfg.get("embed_batch_size", 64)
    fp16 = cfg.get("save_fp16", False)

    for split, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        print(f"\nExtracting {split} ({len(df)} images)...")
        cls, patch = extract_embeddings(model, processor, df, n_reg, batch_size)
        save_split(cls, patch, df, emb_dir, split, fp16)

    print("\nDone.")


if __name__ == "__main__":
    main()
