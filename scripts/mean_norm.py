import torch
import tqdm
from src.config import *
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from src.data.dataset import Dataset, ChallengeTrain
from src.data.data_utils import get_challenge_split


transform_pipeline = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
df_train, _, _, _ = get_challenge_split()
raw_dataset = Dataset(df=df_train, image_dir=IMG_DIR, training=True, transform=transform_pipeline)
standard_dataset = ChallengeTrain(raw_dataset)
train_loader = DataLoader(standard_dataset, batch_size=256, shuffle=False,
                      num_workers=0)
      
mean = torch.zeros(3)
std = torch.zeros(3)
l = 0
p = tqdm.tqdm(train_loader)
for batch in p:
    l+=1
    image_batch = batch[0]
    mean += image_batch.mean(dim=[0,2,3])
    std += image_batch.std(dim=[0,2,3])

mean /= l
std /= l

print("mean by chanel =", mean)
print("Std by chanel =", std)