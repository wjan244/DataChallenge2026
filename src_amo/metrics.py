import numpy as np
import pandas as pd


def error_fn(df:pd.DataFrame)->float:
    pred = df.loc[:, "pred"]
    ground_truth = df.loc[:, "FaceOcclusion"]
    weight = 1/30 + ground_truth

    return np.sum(((pred - ground_truth)**2) * weight, axis=0) / np.sum(weight, axis=0)

def metric_fn(female:pd.DataFrame, male:pd.DataFrame)->float:
    err_male = error_fn(male)
    err_female = error_fn(female)
    return (err_male + err_female) / 2 + abs(err_male - err_female)
