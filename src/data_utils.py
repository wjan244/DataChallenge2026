import numpy as np
import pandas as pd

from scipy.stats import beta, entropy
from sklearn.model_selection import train_test_split
from PIL import Image

from src.config import CSV_DIR, N_SAMPLE

bins = np.linspace(0,1,31)
bin_center = (bins[:-1]+bins[1:])/2
eps = 1e-6

df_train_raw = pd.read_csv(CSV_DIR / "train.csv", delimiter=',')
df_test_raw = pd.read_csv(CSV_DIR / "test_students.csv", delimiter=',')

def _distribution_adaptation(n_sample, df, test_distribution=None):

    # distribution de df
    train_distribution, _ = np.histogram(df["FaceOcclusion"],bins=30, density=True) 
    train_distribution = (train_distribution + eps) / (np.sum(train_distribution)+eps)

    # estimation distribution cible (test)
    if test_distribution is None:
        test_distribution = beta.pdf(bin_center,a=1.5,b=5) + eps
        test_distribution = test_distribution/(np.sum(test_distribution)+eps)   # normalisation

    # calcul du ratio de divergence KL
    ratio_KL = test_distribution / (train_distribution + 1e-6)

    # pondération des occlusion (fonction du ratio KL)
    bin_categorie = pd.cut(df["FaceOcclusion"],bins=bins,include_lowest=True,ordered=True,labels=False)
    image_ponderation = ratio_KL[bin_categorie]
    df['D_KL'] = image_ponderation

    sub_df = df.sample(n=n_sample,weights='D_KL',replace=False,random_state=42)

    sub_df = sub_df.rename(columns={"D_KL": "iw"})

    return sub_df, test_distribution, train_distribution

def get_challenge_split(screenshot_path=None):

    # Remove nan values
    df_train_clean = df_train_raw.dropna()
    df_test = df_test_raw.dropna().reset_index(drop=True)

    # split train et val_samp
    df_train, df_val = train_test_split(df_train_clean,test_size=0.2,random_state=42,shuffle=True)

    df_train = df_train.reset_index(drop=True)
    df_val_raw = df_val.reset_index(drop=True).copy()

    # distribution cible : pixels si screenshot fournie, sinon None → Beta dans _distribution_adaptation
    test_dist = _get_test_distribution_from_screenshot(screenshot_path) if screenshot_path else None

    # adaptation de train et eval à la distribution cible (test)
    n = len(df_train) if screenshot_path else N_SAMPLE
    df_train, _ , _ = _distribution_adaptation(n_sample=n, df=df_train, test_distribution=test_dist)
    df_val_samp, _, _ = _distribution_adaptation(n_sample=5000,df=df_val, test_distribution=test_dist)

    return df_train, df_val_raw, df_val_samp, df_test

def _get_test_distribution_from_screenshot(screenshot_path, n_bins=30):
    """
    screenshot_path : capture d'écran cadrée sur le graphique test uniquement
    (juste la zone des barres, sans axes ni labels)
    """
    arr = np.array(Image.open(screenshot_path).convert("RGB"))
    
    plot_height, plot_width = arr.shape[:2]
    
    # Détection des barres bleues
    blue_mask = (arr[:,:,2].astype(int) - arr[:,:,0].astype(int)) > 5

    # Agréger en n_bins
    bin_counts = np.zeros(n_bins)
    bw = plot_width / n_bins
    for i in range(n_bins):
        s = int(i * bw)
        e = max(s+1, int((i+1) * bw))
    
        # Rectangle de hauteur plot_height, largeur (e-s)
        patch = blue_mask[:, s:e]          # shape: (plot_height, e-s)
        bin_counts[i] = patch.mean()       # proportion de pixels bleus dans le rectangle

    eps = 1e-6
    return (bin_counts + eps) / (bin_counts.sum() + eps * n_bins)


if __name__ == "__main__":
    #tracé de distribution & calcul de la distance KL asscoiée avec test
    
    import matplotlib.pyplot as plt

    df_train_sub, df_val_raw, df_val_samp, df_test = get_challenge_split()

    test_distribution = beta.pdf(bin_center, a=1.5, b=5)
    test_distribution = (test_distribution + eps) / (np.sum(test_distribution) + eps)

    # distribution
        # train
    train_distribution, _ = np.histogram(df_train_raw["FaceOcclusion"],bins=30, density=True) 
    train_distribution = (train_distribution + eps) / (np.sum(train_distribution)+eps)
        # train échantillonné
    train_sub_distribution, _ = np.histogram(df_train_sub["FaceOcclusion"], density=True, bins=bins)
    train_sub_distribution = (train_sub_distribution + eps) / (np.sum(train_sub_distribution) + eps)

        # raw_eval
    val_raw_distribution, _ = np.histogram(df_val_raw["FaceOcclusion"], bins=bins, density=True)
    val_raw_distribution = (val_raw_distribution + eps) / (np.sum(val_raw_distribution) + eps)

        # eval_samp
    val_samp_distribution, _ = np.histogram(df_val_samp["FaceOcclusion"], bins=bins, density=True)
    val_samp_distribution = (val_samp_distribution + eps) / (np.sum(val_samp_distribution) + eps)

    #  divergences KL
    DKL_train_test = entropy(test_distribution,train_distribution,base=None)
    DKL_subtrain_test = entropy(test_distribution, train_sub_distribution,base=None)
    DKL_valraw_test = entropy(test_distribution, val_raw_distribution,base=None)
    DKL_valsamp_test = entropy(test_distribution, val_samp_distribution,base=None)

    # Tracé
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(bin_center, test_distribution, label="Test (Beta)", color="red", linewidth=2)
    ax1.bar(bin_center, train_sub_distribution, width=0.03, alpha=0.5, label=f"Train échantillonné Dkl = {DKL_subtrain_test:.3f}", color="blue")
    ax1.bar(bin_center, train_distribution, width=0.03, alpha=0.5, label=f"Train raw DKL = {DKL_train_test:.3f}", color="green")
    ax1.set_xlabel("Taux d'occlusion")
    ax1.set_title("Transformation de train")
    ax1.legend()

    ax2.plot(bin_center, test_distribution, label="Test (Beta)", color="red", linewidth=2)
    ax2.bar(bin_center, val_samp_distribution, width=0.03, alpha=0.5, label=f"Eval échantillonné DKL = {DKL_valsamp_test:.3f}", color="blue")
    ax2.bar(bin_center, val_raw_distribution, width=0.03, alpha=0.5, label=f"Eval raw DKL = {DKL_valraw_test:.3f}", color="orange")
    ax2.set_xlabel("Taux d'occlusion")
    ax2.set_title("Transformation de eval")
    ax2.legend()

    plt.tight_layout()
    plt.show()






