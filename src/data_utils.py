import numpy as np
import pandas as pd

from scipy.stats import beta

from src.config import CSV_DIR, N_SAMPLE

def distribution_adaptation(n_sample,df):

    # distribution de df
    bins = np.linspace(0,1,31)
    bin_center = (bins[:-1]+bins[1:])/2
    train_distribution, _ = np.histogram(df["FaceOcclusion"],bins=30, density=True)

    # estimation distribution cible (test)
    test_distribution = beta.pdf(bin_center,a=1.5,b=5)
    test_distribution = test_distribution/np.sum(test_distribution)   # normalisation

    # calcul du ratio de divergence KL
    ratio_KL = test_distribution / (train_distribution + 1e-6)

    # pondération des occlusion (fonction du ratio KL)
    bin_categorie = pd.cut(df["FaceOcclusion"],bins=bins,include_lowest=True,ordered=True,labels=False)
    image_ponderation = ratio_KL[bin_categorie]
    df['D_KL'] = image_ponderation

    sub_df = df.sample(n=n_sample,weights='D_KL',replace=False,random_state=42)

    sub_df = sub_df.drop(columns=["D_KL"])

    return sub_df

def get_challenge_split():
    df_train_raw = pd.read_csv(CSV_DIR / "train.csv", delimiter=',')
    df_test_raw = pd.read_csv(CSV_DIR / "test_students.csv", delimiter=',')

    # Remove nan values
    df_train_clean = df_train_raw.dropna()
    df_test = df_test_raw.dropna().reset_index(drop=True)

    # Split Dataframe in train and val
    df_val = df_train_clean.iloc[:20000].reset_index(drop=True)
    df_train = df_train_clean.iloc[20000:].reset_index(drop=True).copy()

    # adaptation de train à la distribution cible (test)
    df_train  = distribution_adaptation(n_sample=N_SAMPLE,df=df_train)

    return df_train, df_val, df_test

