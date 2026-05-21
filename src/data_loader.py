import pandas as pd

from src.config import CSV_DIR

def get_challenge_split():
    df_train_raw = pd.read_csv(CSV_DIR / "train.csv", delimiter=',')
    df_test_raw = pd.read_csv(CSV_DIR / "test_students.csv", delimiter=',')

    # Remove nan values
    df_train_clean = df_train_raw.dropna()
    df_test = df_test_raw.dropna().reset_index(drop=True)

    # Split Dataframe in train and val
    df_val = df_train_clean.iloc[:20000].reset_index(drop=True)
    df_train = df_train_clean.iloc[20000:].reset_index(drop=True)

    return df_train, df_val, df_test