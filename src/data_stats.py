import numpy as np
import pandas as pd

from PIL import Image
from src.path import *
from scipy.stats import beta

bins = np.linspace(0, 1, N_BINS + 1)
bin_center = (bins[:-1] + bins[1:]) / 2
eps = 1e-6


def get_test_distribution_from_screenshot(screenshot_path, n_bins=N_BINS):
    """
    screenshot_path : capture d'écran cadrée sur le graphique test uniquement
    (juste la zone des barres, sans axes ni labels)
    """
    arr = np.array(Image.open(screenshot_path).convert("RGB"))
    _, plot_width = arr.shape[:2]
    
    # détection des barres bleues
    blue_mask = (arr[:, :, 2].astype(int) - arr[:, :, 0].astype(int)) > 5

    # agréger en n_bins
    bin_counts = np.zeros(n_bins)
    bw = plot_width / n_bins
    for i in range(n_bins):
        s = int(i * bw)
        e = max(s + 1, int((i + 1) * bw))
    
        # rectangle de hauteur plot_height, largeur (e-s)
        patch = blue_mask[:, s:e]          # shape: (plot_height, e-s)
        bin_counts[i] = patch.mean()       # proportion de pixels bleus dans le rectangle

    return (bin_counts + eps) / (bin_counts.sum() + eps * n_bins)


def distribution_adaptation_reweight(n_sample, df, test_distribution=None):
    """ Adaptation basée sur la distribution estimée par screenshot (Reweighting) """

    train_distribution, _ = np.histogram(df["FaceOcclusion"], bins=N_BINS, density=True) 
    train_distribution = (train_distribution + eps) / (np.sum(train_distribution) + eps)

    if test_distribution is None:
        test_distribution = beta.pdf(bin_center, a=1.5, b=5) + eps
        test_distribution = test_distribution / (np.sum(test_distribution) + eps)

    ratio_KL = test_distribution / (train_distribution + 1e-6)
    bin_categorie = pd.cut(df["FaceOcclusion"], bins=bins, include_lowest=True, ordered=True, labels=False)
    df['D_KL'] = ratio_KL[bin_categorie]

    sub_df = df.sample(n=n_sample, weights='D_KL', replace=False, random_state=42)
    return sub_df.rename(columns={"D_KL": "iw"}), test_distribution, train_distribution

# def distribution_adaptation_DKL(n_sample:int,df:pd.DataFrame)->tuple[pd.DataFrame,np.ndarray,np.ndarray]:
#     """
#     - représente la distribution test (cible) en une loi beta (1.5,5)
#     - transforme la distribution train en distribution cible par sampling DKL"""

#     # distribution de df
#     train_distribution, _ = np.histogram(df["FaceOcclusion"],bins=30, density=True) 
#     train_distribution = (train_distribution + eps) / (np.sum(train_distribution)+eps)

#     # estimation distribution cible (test)
#     test_distribution = beta.pdf(bin_center,a=1.5,b=5) + eps
#     test_distribution = test_distribution/(np.sum(test_distribution)+eps)   # normalisation

#     # calcul du ratio de divergence KL
#     ratio_KL = test_distribution / (train_distribution + 1e-6)

#     # pondération des occlusion (fonction du ratio KL)
#     bin_categorie = pd.cut(df["FaceOcclusion"],bins=bins,include_lowest=True,ordered=True,labels=False)
#     image_ponderation = ratio_KL[bin_categorie]
#     df['D_KL'] = image_ponderation

#     sub_df = df.sample(n=n_sample,weights='D_KL',replace=False,random_state=42)

#     sub_df = sub_df.drop(columns=["D_KL"])

#     return sub_df, test_distribution, train_distribution