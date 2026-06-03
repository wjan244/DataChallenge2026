import numpy as np
import pandas as pd
import torch

from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from PIL import Image
from src.config import*
from src.config_utils import load_config
from src.data.data_stats import distribution_adaptation_reweight, get_test_distribution_from_screenshot


N_BINS_GENDER = 30
ALPHA_SMOOTH = 50
N_SAMPLE = load_config(CONFIG_DEFAULT).get("globaux", {}).get("N_SAMPLES")
BINS = torch.tensor([0.0000, 0.0333, 0.0667, 0.1000, 0.1333, 0.1667, 0.2000, 0.2333, 0.2667,
        0.3000, 0.3333, 0.3667, 0.4000, 0.4333, 0.4667, 0.5000, 0.5333, 0.5667,
        0.6000, 0.6333, 0.6667, 0.7000, 0.7333, 0.7667, 0.8000, 0.8333, 0.8667,
        0.9000, 0.9333, 0.9667, 1.0000], dtype=torch.float64)

def get_challenge_split(screenshot_path=SCREENSHOT_PATH):
    """
    Pipeline principal : Nettoie les données, effectue le split train/validation
    et adapte les distributions via la méthode de tracé par capture d'écran.
    """
    df_train_raw = pd.read_csv(CSV_DIR / "train.csv", delimiter=',')
    df_test_raw  = pd.read_csv(CSV_DIR / "test_students.csv", delimiter=',')
    df_train_clean = df_train_raw.dropna()
    df_test = df_test_raw.dropna().reset_index(drop=True)

    # extraction de la vraie distribution depuis l'image
    test_dist = get_test_distribution_from_screenshot(screenshot_path)
    
    # split initial stratifié de manière aléatoire (80% Train, 20% Eval)
    df_train, df_val = train_test_split(df_train_clean, test_size=0.2, random_state=42, shuffle=True)
    df_train = df_train.reset_index(drop=True)
    df_val_raw = df_val.reset_index(drop=True).copy()

    n = len(df_train) if screenshot_path else N_SAMPLE
    n_val = len(df_val) if screenshot_path else 5000
    
    # application de l'adaptation de domaine sur le train et la validation
    df_train_reweight, _, _ = distribution_adaptation_reweight(n_sample=n, df=df_train, test_distribution=test_dist)
    df_val_reweight, _, _ = distribution_adaptation_reweight(n_sample=n_val, df=df_val, test_distribution=test_dist)

    return df_train_reweight, df_val_raw, df_val_reweight, df_test


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N_BINS = load_config(CONFIG_DEFAULT).get("globaux", {}).get("N_BINS", 20)
    bins = np.linspace(0, 1, N_BINS + 1)
    bin_center = (bins[:-1] + bins[1:]) / 2
    eps = 1e-6
    df_train_raw = pd.read_csv(CSV_DIR / "train.csv", delimiter=',')

    # Génération du split de données adapté via l'image
    df_train_sub, df_val_raw, df_val_samp, df_test = get_challenge_split(screenshot_path=SCREENSHOT_PATH)

    # Récupération de la distribution cible de référence
    test_distribution = get_test_distribution_from_screenshot(SCREENSHOT_PATH)

    # Calcul des histogrammes réels de contrôle
    train_distribution, _ = np.histogram(df_train_raw["FaceOcclusion"], bins=N_BINS, density=True) 
    train_distribution = (train_distribution + eps) / (np.sum(train_distribution) + eps)
    
    train_sub_distribution, _ = np.histogram(df_train_sub["FaceOcclusion"], bins=bins, density=True)
    train_sub_distribution = (train_sub_distribution + eps) / (np.sum(train_sub_distribution) + eps)

    val_raw_distribution, _ = np.histogram(df_val_raw["FaceOcclusion"], bins=bins, density=True)
    val_raw_distribution = (val_raw_distribution + eps) / (np.sum(val_raw_distribution) + eps)

    val_samp_distribution, _ = np.histogram(df_val_samp["FaceOcclusion"], bins=bins, density=True)
    val_samp_distribution = (val_samp_distribution + eps) / (np.sum(val_samp_distribution) + eps)

    # Calcul des métriques de divergence KL finales
    DKL_train_test = entropy(test_distribution, train_distribution)
    DKL_subtrain_test = entropy(test_distribution, train_sub_distribution)
    DKL_valraw_test = entropy(test_distribution, val_raw_distribution)
    DKL_valsamp_test = entropy(test_distribution, val_samp_distribution)

    # analyse comparative de la réduction de divergence globale
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Diagnostic Train
    ax1.plot(bin_center, test_distribution, label="Cible (Screenshot)", color="red", linewidth=2)
    ax1.bar(bin_center, train_sub_distribution, width=0.03, alpha=0.5, label=f"Train adapté (DKL = {DKL_subtrain_test:.3f})", color="blue")
    ax1.bar(bin_center, train_distribution, width=0.03, alpha=0.5, label=f"Train initial (DKL = {DKL_train_test:.3f})", color="green")
    ax1.set_xlabel("Taux d'occlusion")
    ax1.set_title("Adaptation du jeu d'entraînement")
    ax1.legend()

    # Diagnostic Validation
    ax2.plot(bin_center, test_distribution, label="Cible (Screenshot)", color="red", linewidth=2)
    ax2.bar(bin_center, val_samp_distribution, width=0.03, alpha=0.5, label=f"Eval adaptée (DKL = {DKL_valsamp_test:.3f})", color="blue")
    ax2.bar(bin_center, val_raw_distribution, width=0.03, alpha=0.5, label=f"Eval brute (DKL = {DKL_valraw_test:.3f})", color="orange")
    ax2.set_xlabel("Taux d'occlusion")
    ax2.set_title("Adaptation du jeu de validation")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # visualisation de l'impact des Importance Weights (iw)
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(df_train_raw["FaceOcclusion"], bins=N_BINS, density=True, alpha=0.4, label="Train initial (brut)", color="green")
    
    if "iw" in df_train_sub.columns:
        ax.hist(df_train_sub["FaceOcclusion"], bins=N_BINS, weights=df_train_sub["iw"], density=True, alpha=0.5, label="Train rééquilibré (Poids iw)", color="blue")

    ax.plot(bin_center, test_distribution * N_BINS, color="red", linewidth=2, label="Densité cible (Screenshot)")
    ax.set_xlabel("Taux d'occlusion")
    ax.set_title("Impact de la repondération sur le domaine cible")
    ax.legend()
    plt.show()







def compute_gender_weights(y_all, gender_all):
    """Compute per-bin × gender balancing weights from the full training set.

    y_all, gender_all: 1-D float32 tensors over the full training set.
    Returns w_f, w_m: 1-D float32 tensors of length N_BINS_GENDER.
    """
    device = y_all.device
    bins = BINS.float().to(device)
    bin_idx = (torch.bucketize(y_all.float(), bins, right=False) - 1).clamp(0, N_BINS_GENDER - 1)

    female = (gender_all == 0.0).float()
    male   = (gender_all == 1.0).float()
    n_f = torch.zeros(N_BINS_GENDER, device=device).scatter_add(0, bin_idx, female)
    n_m = torch.zeros(N_BINS_GENDER, device=device).scatter_add(0, bin_idx, male)

    n_total = n_f + n_m
    w_f = (n_total + 2 * ALPHA_SMOOTH) / (2 * (n_f + ALPHA_SMOOTH))
    w_m = (n_total + 2 * ALPHA_SMOOTH) / (2 * (n_m + ALPHA_SMOOTH))
    return w_f, w_m


def lookup_gender_weights(y, gender, w_f, w_m) -> np.float32:
    """Look up precomputed gender bin weight for a single sample.

    y, gender: scalar (float, np.float32, etc.).
    w_f, w_m: 1-D float32 tensors of length N_BINS_GENDER (from Dataset.W_F/W_M).
    Returns: np.float32 — compatible with iw/pi in Dataset.__getitem__.
    """
    y_t = torch.tensor([float(y)], dtype=torch.float32)
    bin_idx = (torch.bucketize(y_t, BINS.float(), right=False) - 1).clamp(0, N_BINS_GENDER - 1)
    w = w_f[bin_idx] if float(gender) == 0.0 else w_m[bin_idx]
    return np.float32(w.item())
