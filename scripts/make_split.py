"""Split train.csv into train_split.csv and val_split.csv.

Adds a `noisy` column (1 if the row is in validation_noisy.csv, 0 otherwise).

Two modes:
  - --val-csv PATH  : val_split is exactly the images in PATH; train_split is everything else.
  - (no --val-csv)  : random 80/20 split controlled by --seed and --val-size.

Usage:
    python scripts/make_split.py --val-csv amo/fine_tune_val_preds.csv
    python scripts/make_split.py [--seed 42] [--val-size 0.2] [--out-dir data/occlusion_datasets]
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--val-csv", type=Path, default=None,
                        help="CSV whose filenames define the val split exactly")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "data" / "occlusion_datasets")
    args = parser.parse_args()

    data_dir = ROOT / "data" / "occlusion_datasets"
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(data_dir / "train.csv").dropna().reset_index(drop=True)
    df_noisy = pd.read_csv(data_dir / "validation_noisy.csv").dropna().reset_index(drop=True)

    noisy_filenames = set(df_noisy["filename"])
    df_train["noisy"] = df_train["filename"].isin(noisy_filenames).astype(int)

    if args.val_csv is not None:
        val_csv_path = args.val_csv if args.val_csv.is_absolute() else ROOT / args.val_csv
        df_val_ref = pd.read_csv(val_csv_path)
        val_filenames = set(df_val_ref["filename"])
        val_split  = df_train[df_train["filename"].isin(val_filenames)].reset_index(drop=True)
        train_split = df_train[~df_train["filename"].isin(val_filenames)].reset_index(drop=True)
        missing = val_filenames - set(df_train["filename"])
        if missing:
            print(f"WARNING: {len(missing)} filenames from --val-csv not found in train.csv")
    else:
        train_split, val_split = train_test_split(
            df_train, test_size=args.val_size, random_state=args.seed, shuffle=True
        )
        train_split = train_split.reset_index(drop=True)
        val_split   = val_split.reset_index(drop=True)

    train_path = out_dir / "train_split.csv"
    val_path   = out_dir / "val_split.csv"
    train_split.to_csv(train_path, index=False)
    val_split.to_csv(val_path, index=False)

    print(f"train_split : {len(train_split):>7,} rows  ({train_split['noisy'].sum()} noisy)  → {train_path}")
    print(f"val_split   : {len(val_split):>7,} rows  ({val_split['noisy'].sum()} noisy)  → {val_path}")
    print(val_split["FaceOcclusion"].mean(),df_val_ref["gt"].mean())

if __name__ == "__main__":
    main()
