import os
import math
import torch
import numpy as np
import random
import glob

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_ = None, mode = 'train'):
        
        self.transform = transforms.Compose(transforms_)
        
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        if mode == 'train':
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))
            
    def __getitem__(self, index):
        img = Image.open(self.files[index%len(self.files)])
        
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)
