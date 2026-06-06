import numpy as np
import pandas as pd

from . import df_train_raw,df_test_raw,N_BINS,N_SAMPLE,eps
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from PIL import Image
from src.config import*
from src.config_utils import load_config
from src.data.data_stats import distribution_adaptation_reweight, get_test_distribution_from_screenshot

bins = np.linspace(0, 1, N_BINS + 1)
bin_center = (bins[:-1] + bins[1:]) / 2

def get_challenge_split(screenshot_path=SCREENSHOT_PATH):
    """
    nettoie les données, effectue le split train/validation 
    et adapte les distributions.
    """
    df_train_clean = df_train_raw.dropna()
    df_test = df_test_raw.dropna().reset_index(drop=True)
    
    # split initial stratifié de manière aléatoire (80% Train, 20% Eval)
    df_train, df_val = train_test_split(df_train_clean, test_size=0.2, random_state=42, shuffle=True)
    df_train = df_train.reset_index(drop=True)
    df_val_raw = df_val.reset_index(drop=True).copy()

    n_val = int((0.2*N_SAMPLE))

    # application de l'adaptation de domaine sur le train et la validation
    df_train_reweight, _, _ = distribution_adaptation_reweight(n_sample=N_SAMPLE, df=df_train, test_distribution=None) # forcer la distribution test en une distribution beta
    df_val_reweight, _, _ = distribution_adaptation_reweight(n_sample=n_val, df=df_val, test_distribution=None) # forcer la distribution test en une distribution beta

    return df_train_reweight, df_val_raw, df_val_reweight, df_test


if __name__ == "__main__":
    # vérification des distributions
    import matplotlib.pyplot as plt

    # Génération du split de données adapté via l'image
    df_train_sub, df_val_raw, df_val_samp, df_test = get_challenge_split(screenshot_path=SCREENSHOT_PATH)
    print(f"dimension de train{len(df_train_sub)}")
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
