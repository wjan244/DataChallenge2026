"""
Runs inference with a trained checkpoint on the train or val split and saves a predictions CSV.

Usage:
    python scripts/eval_on_split.py \
        --config vit_tiny_patch16_224.yaml \
        --stage  probing_training \
        --timestamp 2026-05-30_14-32-01 \
        --split  train          # train | val

Output CSV columns: filename, pred, FaceOcclusion, gender
Saved to: submission/<timestamp>_<model>_<stage>_<split>_preds.csv
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import pandas as pd
from tqdm import tqdm

from src.config import DEVICE, CHECKPOINT_DIR, SUBMISSION_DIR, NUM_WORKERS
from src.config_utils import load_config
from src.models.models import get_model
from src.data.data_loader import get_challenge_train_loader, get_challenge_val_loader
from src.metrics import metric_fn


STAGE_TO_LOADER = {
    "domain_adaptation": None,          # CelebA — not applicable here
    "probing_training":  None,
    "lora_training":     None,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on train or val split.")
    parser.add_argument("--config",    required=True,
                        help="YAML filename inside config/models/ (e.g. vit_tiny_patch16_224.yaml)")
    parser.add_argument("--stage",     required=True,
                        choices=["domain_adaptation", "probing_training", "lora_training"],
                        help="Training stage whose checkpoint to load")
    parser.add_argument("--timestamp", required=True,
                        help="Timestamp string from the checkpoint filename")
    parser.add_argument("--split",     default="val",
                        choices=["train", "val"],
                        help="Which split to run inference on (default: val)")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()

    # --- load config ---
    cfg = load_config(args.config)
    cfg_glob   = cfg["globaux"]
    cfg_method = cfg[f"{args.stage}_training"] if f"{args.stage}_training" in cfg else cfg[args.stage]
    model_name = cfg["model"]
    method_kwargs = cfg_method.get("method_kwargs", {})

    model_tag       = f"{model_name}_{args.stage}"
    checkpoint_path = CHECKPOINT_DIR / f"{args.timestamp}_{model_tag}.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Available checkpoints:\n" +
            "\n".join(f"  {p.name}" for p in sorted(CHECKPOINT_DIR.glob("*.pt")))
        )

    # --- model ---
    model = get_model(model_name, num_classes=1, method=args.stage, **method_kwargs)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path.name}")

    # --- data ---
    if args.split == "train":
        loader = get_challenge_train_loader(
            batch_size=args.batch_size,
            model_name=model_name,
            augmentation=False,
        )
    else:
        loader = get_challenge_val_loader(
            split="val_samp",
            batch_size=args.batch_size,
            model_name=model_name,
        )

    # --- inference ---
    results = []
    with torch.inference_mode():
        for X, y, gender, filename, *_ in tqdm(loader, desc=f"inference ({args.split})"):
            X = X.to(DEVICE)
            y_pred = model(X)
            for i in range(len(X)):
                results.append({
                    "filename":      filename[i],
                    "pred":          float(y_pred[i]),
                    "FaceOcclusion": float(y[i]),
                    "gender":        float(gender[i]),
                })

    df = pd.DataFrame(results)

    # --- competition metric ---
    males   = df[df["gender"] == 1.0]
    females = df[df["gender"] == 0.0]
    score   = metric_fn(females, males)
    print(f"\nCompetition score on {args.split}: {score:.6f}")

    # --- save ---
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    out = SUBMISSION_DIR / f"{args.timestamp}_{model_tag}" / f"{args.split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows → {out}")


if __name__ == "__main__":
    main()
