# from datetime import datetime
# from train import run_train

# if __name__ == '__main__':
#     timestamp = f"{datetime.now():%Y-%m-%d_%H:%M}"
#     run_train(timestamp)


from datetime import datetime
from train import run_train
from src.data_utils import get_challenge_split
from src.config import SCREENSHOT_PATH

if __name__ == '__main__':
    timestamp = f"{datetime.now():%Y-%m-%d_%H:%M}"
    
    df_train, df_val_raw, df_val_samp, df_test = get_challenge_split(screenshot_path=SCREENSHOT_PATH)
    
    w = df_train["iw"]
    n_eff = w.sum()**2 / (w**2).sum()
    print(f"n effectif : {n_eff:.0f} / {len(df_train)} ({100*n_eff/len(df_train):.1f}%)")
