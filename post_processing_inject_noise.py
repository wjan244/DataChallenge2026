"""
post_processing_inject_noise.py — Injection d'un bruit gaussien sur les
                                   prédictions d'occlusion du genre ESTIMÉ (par DINOv3 + LR).

Pipeline :
  1. CLS de DINOv3 (pré-entraîné) sur train (sous-échantillon), val, test.
     → cache parquet dans cache_cls/. Chaque extraction ne tourne QUE si
       son parquet est absent.
  2. Régression logistique BALANCED  train CLS → genre.
  3. Application au TEST → genre estimé par image  → test_genre.csv
  4. Confusion estimé/vrai sur la VAL (seul split avec vrai genre côté inférence)
     → confusion_matrix.csv
  5. Injection N(0, var) sur les prédictions d'occlusion des images du test
     PRÉDITES == genre cible.  PAS DE CLIPPING.
     → test_{genre}_{var}.csv

Entrées :
  --test-csv   : CSV de soumission produit par train_dino_trainval.py
                 (colonnes : filename, FaceOcclusion, gender)
  --genre      : genre cible où injecter le bruit  ('F' ou 'M')
  --var        : variance du bruit gaussien (l'écart-type injecté = sqrt(var))

Lancer depuis la racine du projet :
    python post_processing_inject_noise.py \
        --test-csv data/results/raw_prediction.csv \
        --genre F --var 0.0025
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from src.config import IMG_DIR, NUM_WORKERS
from src.data.data_utils import get_challenge_split
from src.data.dataset import Dataset as ChallengeDataset

# ==============================================================================
# CONFIG
# ==============================================================================

MODEL_NAME    = "vit_base_patch16_dinov3.lvd1689m"
CACHE_DIR     = Path("cache_cls")
OUT_DIR       = Path("data/results")               # sorties de ce script
TRAIN_SUBSAMP = 40000                              # sous-échantillon train pour la LR
BATCH_SIZE    = 128
SEED          = 42

# Convention de codage du genre (cohérente avec le reste du repo) :
#   0.0 = Femme (F)   |   1.0 = Homme (M)
GENDER_CODE = {"F": 0.0, "M": 1.0}

# ==============================================================================
# DEVICE / SEED
# ==============================================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Device : {DEVICE}")
torch.manual_seed(SEED)
np.random.seed(SEED)

# ==============================================================================
# EXTRACTION CLS (avec cache)
# ==============================================================================

def build_backbone_and_transform():
    backbone  = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0).to(DEVICE).eval()
    cfg       = timm.data.resolve_model_data_config(backbone)
    transform = timm.data.create_transform(**cfg, is_training=False)   # déterministe
    return backbone, transform


def extract_cls(df, backbone, transform, tag, training):
    """
    Renvoie (et cache) un DataFrame des CLS de DINOv3 pour les images de df.
    CLS = token d'indice 0 de forward_features.

    training=True  : dataset renvoie (X, y, gender, filename, iw, pi)
                     → on récupère gender (utile pour train et val).
    training=False : dataset renvoie (X, filename)
                     → pas de gender disponible (cas du test).

    Le parquet n'est (re)calculé QUE s'il n'existe pas déjà.
    """
    cache = CACHE_DIR / f"cls_{tag}.parquet"
    if cache.exists():
        print(f"  CLS '{tag}' chargés depuis le cache : {cache}")
        return pd.read_parquet(cache)

    print(f"  CLS '{tag}' : extraction ({len(df)} images)...")
    dataset = ChallengeDataset(df, IMG_DIR, training=training, transform=transform)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
    )

    filenames, genders, cls_list = [], [], []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"  CLS {tag}", leave=False):
            X = batch[0].to(DEVICE)
            if training:
                gender   = batch[2]
                filename = batch[3]
                genders.append(np.asarray(gender, dtype=np.float32))
            else:
                filename = batch[1]
            feats = backbone.forward_features(X)        # (B, N, D)
            cls   = feats[:, 0, :].cpu().numpy()        # (B, D) — token CLS
            cls_list.append(cls)
            filenames.extend(list(filename))

    cls_arr = np.concatenate(cls_list, axis=0)
    out = pd.DataFrame(cls_arr, columns=[f"cls_{i}" for i in range(cls_arr.shape[1])])
    out.insert(0, "filename", filenames)
    if training:
        out.insert(1, "gender", np.concatenate(genders))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    print(f"  CLS '{tag}' sauvegardés : {cache}  ({len(out)} lignes)")
    return out

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", type=Path, required=True,
                        help="CSV de soumission (filename, FaceOcclusion, gender)")
    parser.add_argument("--genre", type=str, required=True, choices=["F", "M"],
                        help="genre cible où injecter le bruit")
    parser.add_argument("--var", type=float, required=True,
                        help="variance du bruit gaussien (sigma = sqrt(var))")
    args = parser.parse_args()

    target_code = GENDER_CODE[args.genre]
    sigma       = float(np.sqrt(args.var))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Données + backbone ----------------------------------------------
    print("\n--- Split challenge ---")
    df_train, df_val_raw, _, df_test = get_challenge_split()
    print(f"  train : {len(df_train)}  |  val : {len(df_val_raw)}  |  test : {len(df_test)}")

    backbone, transform = build_backbone_and_transform()

    # ---- 2. CLS (train / val / test) avec cache ------------------------------
    print("\n--- Extraction CLS (cache_cls/) ---")
    df_train_sub = df_train.sample(n=min(TRAIN_SUBSAMP, len(df_train)), random_state=SEED)
    cls_train = extract_cls(df_train_sub, backbone, transform, tag="train", training=True)
    cls_val   = extract_cls(df_val_raw,   backbone, transform, tag="val",   training=True)
    cls_test  = extract_cls(df_test,      backbone, transform, tag="test",  training=False)

    feat_cols = [c for c in cls_train.columns if c.startswith("cls_")]

    # ---- 3. LR genre BALANCED ------------------------------------------------
    print("\n--- Régression logistique genre (balanced) ---")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    clf.fit(cls_train[feat_cols].values, cls_train["gender"].values)

    train_acc = clf.score(cls_train[feat_cols].values, cls_train["gender"].values)
    val_acc   = clf.score(cls_val[feat_cols].values,   cls_val["gender"].values)
    print(f"  accuracy genre — train : {train_acc:.4f}  |  val : {val_acc:.4f}")

    # ---- 4. Confusion sur la VAL (vrai genre dispo) --------------------------
    # Convention : 0.0 = F, 1.0 = M.
    # Lignes = vrai genre, colonnes = genre prédit.
    print("\n--- Matrice de confusion (val) ---")
    gender_true_val = cls_val["gender"].values
    gender_est_val  = clf.predict(cls_val[feat_cols].values)

    def counts(true_code, pred_code):
        return int(np.sum((gender_true_val == true_code) & (gender_est_val == pred_code)))

    n_FF = counts(0.0, 0.0)   # vraie F prédite F
    n_FM = counts(0.0, 1.0)   # vraie F prédite M
    n_MF = counts(1.0, 0.0)   # vrai  M prédit  F
    n_MM = counts(1.0, 1.0)   # vrai  M prédit  M

    conf = pd.DataFrame(
        [[n_FF, n_FM], [n_MF, n_MM]],
        index=["true_F", "true_M"],
        columns=["pred_F", "pred_M"],
    )
    conf_path = OUT_DIR / "confusion_matrix.csv"
    conf.to_csv(conf_path)
    print(conf)

    # Indicateurs utiles pour la calibration du bruit (du point de vue 'F') :
    #   Recall(F)   = FF / (FF + FM)
    #   FPR(F)      = MF / (MF + MM)      (vrais M classés F, rapportés aux vrais M)
    recall_F = n_FF / (n_FF + n_FM + 1e-12)
    fpr_F    = n_MF / (n_MF + n_MM + 1e-12)
    print(f"  Recall(F) = {recall_F:.4f}   FPR(F) = {fpr_F:.4f}   "
          f"levier (R - FPR) = {recall_F - fpr_F:.4f}")
    print(f"  Confusion sauvegardée : {conf_path}")

    # ---- 5. Genre estimé sur le TEST  → test_genre.csv -----------------------
    print("\n--- Prédiction du genre sur le test ---")
    gender_est_test = clf.predict(cls_test[feat_cols].values)
    # remappe 0.0/1.0 → 'F'/'M' pour lisibilité
    inv_code = {v: k for k, v in GENDER_CODE.items()}
    df_genre = pd.DataFrame({
        "filename":   cls_test["filename"].values,
        "gender_est": [inv_code[g] for g in gender_est_test],
    })
    genre_path = OUT_DIR / "test_genre.csv"
    df_genre.to_csv(genre_path, index=False)
    n_estF = int(np.sum(gender_est_test == 0.0))
    n_estM = int(np.sum(gender_est_test == 1.0))
    print(f"  test prédit : {n_estF} F  |  {n_estM} M")
    print(f"  Genre estimé sauvegardé : {genre_path}")

    # ---- 6. Injection du bruit sur le genre cible (AVEC clipping) ------------
    print(f"\n--- Injection bruit : genre={args.genre}  var={args.var}  (sigma={sigma:.5f}) ---")
    df_sub = pd.read_csv(args.test_csv)
    assert "FaceOcclusion" in df_sub.columns and "filename" in df_sub.columns, \
        "Le CSV de soumission doit contenir 'filename' et 'FaceOcclusion'."

    # genre estimé aligné sur le CSV de soumission via filename
    est_map = dict(zip(cls_test["filename"].values, gender_est_test))
    sub_est = df_sub["filename"].map(est_map)
    n_missing = int(sub_est.isna().sum())
    if n_missing:
        print(f"  ⚠ {n_missing} filenames du CSV sans genre estimé "
              f"(non bruités). Vérifie que le test.csv et le test du split coïncident.")

    target_mask = (sub_est == target_code).fillna(False).values

    # bruit gaussien centré, variance = args.var, UNIQUEMENT sur les prédites 'genre'
    rng   = np.random.default_rng(SEED)
    noise = rng.normal(loc=0.0, scale=sigma, size=len(df_sub))

    occ = df_sub["FaceOcclusion"].to_numpy(dtype=float).copy()
    occ[target_mask] = occ[target_mask] + noise[target_mask]
    occ = np.clip(occ, 0, 1)  # AVEC clip(0,1)

    df_out = df_sub.copy()
    df_out["FaceOcclusion"] = occ

    out_path = OUT_DIR / f"test_{args.genre}_{args.var}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"  {int(target_mask.sum())} images bruitées (prédites {args.genre}).")
    print(f"  Soumission bruitée sauvegardée : {out_path}")


if __name__ == "__main__":
    main()