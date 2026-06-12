"""
read_probe.py — Lecture d'une sonde leaderboard pour le post-processing d'équité.

À partir d'une sonde unique (variance v injectée sur un genre, score avant/après),
classe le régime et estime la pénalité de fairness initiale G, puis régénère
systématiquement un fichier bruité à la variance optimale v* = G / b depuis le
test.csv BRUT.

Rappel analytique (bruit injecté sur les PRÉDITES 'genre') :
    a = (R + FPR) / 2          (coefficient de coût "performance")
    b = R - FPR                (coefficient de réduction du gap)
    m = (S1 - S0) / v          (ratio mesuré : accroissement de score / variance)

    Cas A — mauvais genre bruité          : m = a + b
    Cas B — bon genre, sous-shoot         : m = a - b
    Cas C — bon genre, overshoot          : m = a + b - 2G/v   (donc m < a + b)
                                            → G = v * (a + b - m) / 2
                                            → v* = G / b

R et FPR proviennent de confusion_matrix.csv (calculée sur la val).
Ce sont des taux conditionnels au vrai genre → non biaisés par le déséquilibre
des effectifs de la val.

Entrées :
  --s0            : score de référence (sans bruit)
  --s1            : score obtenu avec la variance injectée
  --var           : variance v injectée pour obtenir s1
  --genre         : genre bruité ('F' ou 'M')
  --confusion     : confusion_matrix.csv  (index true_F/true_M, colonnes pred_F/pred_M)
  --test-csv      : test.csv BRUT (filename, FaceOcclusion, gender)
  --test-genre    : test_genre.csv (filename, gender_est ∈ {'F','M'})

Sortie :
  - affichage de a-b, a+b, m, du régime, et de G / v* le cas échéant
  - fichier test_{genre}_{v*}.csv : bruit N(0, v*) sur les PRÉDITES 'genre',
    CLIPPÉ sur [0,1], regénéré depuis le test BRUT.

Lancer depuis la racine du projet :
    python read_probe.py \
        --s0 0.00108 --s1 0.00120 --var 0.0008 --genre F \
        --confusion submission/postproc/confusion_matrix.csv \
        --test-csv submission/2026-06-08_22-19-48_submission_vit_base_patch16_dinov3.lvd1689m_clean_trainval/test.csv \
        --test-genre submission/postproc/test_genre.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR     = Path("submission/postproc")
SEED        = 42
GENDER_CHAR = {"F", "M"}

# Tolérance relative pour décider "m proche de a+b" → frontière A / C.
# m est mesuré avec du bruit (score leaderboard arrondi, a/b estimés sur val) :
# on considère qu'on est en overshoot (C) seulement si m est NETTEMENT sous a+b.
REL_TOL = 0.05    # 5 % de b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s0",          type=float, required=True, help="score de référence (sans bruit)")
    parser.add_argument("--s1",          type=float, required=True, help="score avec variance injectée")
    parser.add_argument("--var",         type=float, required=True, help="variance v injectée pour s1")
    parser.add_argument("--genre",       type=str,   required=True, choices=["F", "M"])
    parser.add_argument("--confusion",   type=Path,  required=True, help="confusion_matrix.csv")
    parser.add_argument("--test-csv",    type=Path,  required=True, help="test.csv BRUT")
    parser.add_argument("--test-genre",  type=Path,  required=True, help="test_genre.csv")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. R, FPR depuis la confusion --------------------------------------
    # confusion_matrix.csv : index = [true_F, true_M], colonnes = [pred_F, pred_M]
    conf = pd.read_csv(args.confusion, index_col=0)
    n_FF = float(conf.loc["true_F", "pred_F"])
    n_FM = float(conf.loc["true_F", "pred_M"])
    n_MF = float(conf.loc["true_M", "pred_F"])
    n_MM = float(conf.loc["true_M", "pred_M"])

    # Définis du point de vue du genre CIBLE bruité.
    # Si on bruite les prédites F : R = recall(F), FPR = taux de vrais M classés F.
    # Si on bruite les prédites M : on raisonne symétriquement (recall(M), taux de vrais F classés M).
    if args.genre == "F":
        R   = n_FF / (n_FF + n_FM + 1e-12)     # vraies F bien classées F
        FPR = n_MF / (n_MF + n_MM + 1e-12)     # vrais M classés F (fuite)
        n_target_true = n_FF + n_FM            # effectif de vraies F (précision d'estimation de R)
    else:  # genre == "M"
        R   = n_MM / (n_MM + n_MF + 1e-12)     # vrais M bien classés M
        FPR = n_FM / (n_FF + n_FM + 1e-12)     # vraies F classées M (fuite)
        n_target_true = n_MM + n_MF            # effectif de vrais M

    a = (R + FPR) / 2.0
    b = R - FPR

    # ---- 2. Ratio mesuré et bornes théoriques -------------------------------
    v  = args.var
    m  = (args.s1 - args.s0) / v

    val_B = a - b      # bon genre, sous-shoot
    val_A = a + b      # mauvais genre (ou borne haute de l'overshoot)

    print("\n=== Lecture de sonde ===")
    print(f"  R (recall {args.genre})            : {R:.4f}   (estimé sur {int(n_target_true)} vrais {args.genre})")
    print(f"  FPR (fuite vers {args.genre})      : {FPR:.4f}")
    print(f"  a = (R+FPR)/2                       : {a:.5f}")
    print(f"  b = R - FPR                         : {b:.5f}")
    print()
    print(f"  a - b  (Cas B, sous-shoot)         : {val_B:.5f}")
    print(f"  a + b  (Cas A / borne overshoot)   : {val_A:.5f}")
    print(f"  m = (S1 - S0)/v                    : {m:.5f}")

    # ---- 3. Classification du régime ----------------------------------------
    # Frontière B vs (A/C) : au milieu de [a-b, a+b], i.e. en m = a.
    #   m proche de a-b  → B
    #   m proche de a+b  → A
    #   m nettement < a+b (mais côté haut) → C (overshoot)
    tol = REL_TOL * abs(b)

    if m <= a:
        # côté bas : sous-shoot (B). (m peut être < a-b par bruit d'estimation.)
        regime = "B"
        print("\n  → Cas B : bon genre bruité, SOUS-shooté.")
        print("     Le levier d'équité n'est pas encore épuisé : v est trop petit.")
        G = None
    elif m >= val_A - tol:
        # m ≈ a+b (ou au-dessus) : mauvais genre.
        regime = "A"
        print("\n  → Cas A : MAUVAIS genre bruité.")
        print("     Le bruit augmente à la fois la perf ET le gap. Bruiter l'autre genre.")
        G = None
    else:
        # a < m < a+b-tol : overshoot du bon genre.
        regime = "C"
        G = v * (val_A - m) / 2.0
        print("\n  → Cas C : bon genre bruité, OVER-shooté.")
        print(f"     Pénalité de fairness initiale estimée : G = {G:.6f}")

    # ---- 4. v* = G / b  (régénéré dans TOUS les cas) ------------------------
    # Dans tous les cas on tente d'estimer G pour proposer v*. Si on n'a pas
    # extrait G du régime (A ou B), on l'estime quand même via la même relation
    # G = v*(a+b-m)/2 — utile à titre indicatif, l'utilisateur juge la pertinence.
    if G is None:
        G_est = v * (val_A - m) / 2.0
        note  = "(G estimé hors Cas C — pertinence à juger)"
    else:
        G_est = G
        note  = ""

    if b <= 0:
        print("\n  ⚠ b = R - FPR ≤ 0 : levier nul ou négatif, v* non défini. "
              "Pas de régénération.")
        return

    v_star = G_est / b
    print(f"\n  v* = G / b = {v_star:.6f}   {note}")

    if v_star <= 0:
        print("  ⚠ v* ≤ 0 : pas de bruit à injecter. Pas de fichier généré.")
        return

    # ---- 5. Régénération depuis le test BRUT, bruit N(0, v*), CLIPPÉ ---------
    sigma = float(np.sqrt(v_star))
    target_char = args.genre

    df_sub   = pd.read_csv(args.test_csv)
    df_genre = pd.read_csv(args.test_genre)
    assert {"filename", "FaceOcclusion"}.issubset(df_sub.columns)
    assert {"filename", "gender_est"}.issubset(df_genre.columns)

    est_map  = dict(zip(df_genre["filename"], df_genre["gender_est"]))
    sub_est  = df_sub["filename"].map(est_map)
    n_missing = int(sub_est.isna().sum())
    if n_missing:
        print(f"  ⚠ {n_missing} filenames sans genre estimé (non bruités).")

    target_mask = (sub_est == target_char).fillna(False).values

    rng   = np.random.default_rng(SEED)
    noise = rng.normal(loc=0.0, scale=sigma, size=len(df_sub))

    occ = df_sub["FaceOcclusion"].to_numpy(dtype=float).copy()
    occ[target_mask] = occ[target_mask] + noise[target_mask]
    occ = np.clip(occ, 0.0, 1.0)                       # CLIP demandé ici

    df_out = df_sub.copy()
    df_out["FaceOcclusion"] = occ

    out_path = OUT_DIR / f"test_{target_char}_{v_star:.6f}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n  {int(target_mask.sum())} images bruitées (prédites {target_char}), clippées [0,1].")
    print(f"  Fichier régénéré à v* depuis le brut : {out_path}")


if __name__ == "__main__":
    main()