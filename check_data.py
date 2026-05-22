from PIL import Image
from tqdm import tqdm

from src.config import IMG_DIR
from src.data_utils import get_challenge_split

if __name__ == "__main__":

    df_train, df_val, df_test = get_challenge_split()
        
    for idx, row in tqdm(df_train.iterrows(), total=len(df_train)):
        try:
            filename = df_train.loc[idx, 'filename']
            img2display = Image.open(IMG_DIR/ filename)
        except ValueError as e:
            print(idx, e)

    for idx, row in tqdm(df_val.iterrows(), total=len(df_val)):
        try:
            filename = df_val.loc[idx, 'filename']
            img2display = Image.open(IMG_DIR/ filename)
        except ValueError as e:
            print(idx, e)
            
    for idx, row in tqdm(df_test.iterrows(), total=len(df_test)):
        try:
            filename = df_test.loc[idx, 'filename']
            img2display = Image.open(IMG_DIR/ filename)
        except ValueError as e:
            print(idx, e)