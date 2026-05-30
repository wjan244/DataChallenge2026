import numpy as np
import pandas as pd


def error_fn(df: pd.DataFrame, w=None) -> float:
    pred = df.loc[:, "pred"].astype(float).to_numpy()
    ground_truth = df.loc[:, "FaceOcclusion"].astype(float).to_numpy()

    if w is None:
        weight = 1/30 + ground_truth
        return np.sum(((pred - ground_truth)**2) * weight, axis=0) / np.sum(weight, axis=0)
    else:
        return np.sum(((pred - ground_truth)**2) * w, axis=0) / np.sum(w, axis=0)

def metric_fn(female: pd.DataFrame, male: pd.DataFrame, w=None) -> float:
    if w is None:
        err_female = error_fn(female, None)
        err_male = error_fn(male, None)
    else:
        if isinstance(w, (list, tuple)) and len(w) == 2:
            w_female, w_male = w
            err_female = error_fn(female, w_female)
            err_male = error_fn(male, w_male)
        else:
            err_female = error_fn(female, w)
            err_male = error_fn(male, w)

    return float((err_male + err_female) / 2 + abs(err_male - err_female))

