from __future__ import print_function, division
import os
import torch
from PIL import Image
from glob import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class DeepFakeDatset(Dataset):

    def __init__(self, path, set = 'train', transform = None):
        self.path = path

        self.transform = transform

        if set == 'train' or set == 'val' or set == 'test':
            self.fake_path = path + '/' + set + '/fake'
            self.real_path = path + '/' + set + '/real' 
        else:
            assert set == 'train' or set == 'val' or set == 'test', 'set name must be train, val, test'

        self.real_list = glob(os.path.join(self.real_path, '**/*.png'))
        self.fake_list = glob(os.path.join(self.fake_path, '**/*.png'))

        # fake 1, real 0
        self.img_list = self.real_list + self.fake_list
        self.target_list = [0]*len(self.real_list) + [1]*len(self.fake_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        target = self.target_list[idx]

        data = Image.open(img_path).convert('RGB')

        if not self.transform == None:
            data = self.transform(data)
        
        return data, target


if __name__ == "__main__":
    data_path = "/media/data2/eunju/df_datasets/DeepFake"

    transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    dataset = DeepFakeDatset(data_path, set = 'train', transform = transform)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, drop_last=False)
    
    for data, target in dataloader:
        print(data.shape)
        print(target)
        print(target.shape)
        break
