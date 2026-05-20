import numpy as np 
import torch
from PIL import Image
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, df, image_dir, training=True, transform=None):
         'Initialization'
         self.training = training
         self.image_dir = image_dir
         self.df = df
         self.transform = transform if transform else transforms.ToTensor()
         
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        row = self.df.iloc[index]
        filename = row['filename']

        # Load data and get label
        img_path = self.image_dir / filename
        img = Image.open(img_path).convert('RGB')

        X = self.transform(img)

        if self.training:
            y = row['FaceOcclusion']
            y = np.float32(y)
            gender = row['gender']
            return X, y, gender, filename
        else:
            y = None
            gender = None
            return X, filename