import numpy as np
import pandas as pd

from scipy.stats import beta

bins = np.linspace(0,1,31)
bin_center = (bins[:-1]+bins[1:])/2
eps = 1e-6

def distribution_adaptation_DKL(n_sample:int,df:pd.DataFrame)->tuple[pd.DataFrame,np.ndarray,np.ndarray]:
    """
    - représente la distribution test (cible) en une loi beta (1.5,5)
    - transforme la distribution train en distribution cible par sampling DKL"""

    # distribution de df
    train_distribution, _ = np.histogram(df["FaceOcclusion"],bins=30, density=True) 
    train_distribution = (train_distribution + eps) / (np.sum(train_distribution)+eps)

    # estimation distribution cible (test)
    test_distribution = beta.pdf(bin_center,a=1.5,b=5) + eps
    test_distribution = test_distribution/(np.sum(test_distribution)+eps)   # normalisation

    # calcul du ratio de divergence KL
    ratio_KL = test_distribution / (train_distribution + 1e-6)

    # pondération des occlusion (fonction du ratio KL)
    bin_categorie = pd.cut(df["FaceOcclusion"],bins=bins,include_lowest=True,ordered=True,labels=False)
    image_ponderation = ratio_KL[bin_categorie]
    df['D_KL'] = image_ponderation

    sub_df = df.sample(n=n_sample,weights='D_KL',replace=False,random_state=42)

    sub_df = sub_df.drop(columns=["D_KL"])

    return sub_df, test_distribution, train_distribution