import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
from matplotlib import pyplot as plt
import random


class Zhang(torch.utils.data.Dataset):
    def __init__(self, root, train=True, img_size=(256, 256), normalize=False, enable_transform=True, full=True):

        self.data = []
        self.train = train
        self.root = root
        self.normalize = normalize
        self.img_size = img_size
        self.mean = 0.1307
        self.std = 0.3081
        self.full = full

        if train:
            if enable_transform:
                self.transforms = transforms.Compose([
                    transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.95,1.05)),
                    transforms.ToTensor()
                ])
            else:
                self.transforms = transforms.ToTensor()
        else:
            self.transforms = transforms.ToTensor()

        self.load_data()

    def load_data(self):
        if self.train:
            items = os.listdir(os.path.join(self.root, 'normal_256'))
            for item in items:
                self.data.append((Image.open(os.path.join(self.root, 'normal_256', item)).resize(self.img_size), 0))
        if not self.train:
            items = os.listdir(os.path.join(self.root, 'normal_256'))
            for idx, item in enumerate(items):
                if not self.full and idx > 9:
                    break
                self.data.append((Image.open(os.path.join(self.root, 'normal_256', item)).resize(self.img_size), 0))
            items = os.listdir(os.path.join(self.root, 'pneumonia_256'))
            for idx, item in enumerate(items):
                if not self.full and idx > 9:
                    break
                self.data.append((Image.open(os.path.join(self.root, 'pneumonia_256', item)).resize(self.img_size), 1))
        print('%d data loaded from: %s' % (len(self.data), self.root))
    

    def __getitem__(self, index):
        img, label = self.data[index]

        img = self.transforms(img)[[0]]
        if self.normalize:
            img -= self.mean
            img /= self.std
        return img, (torch.zeros((1,)) + label).long()

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    dataset = Zhang('/media/administrator/1305D8BDB8D46DEE/jhu/ZhangLabData/CellData/chest_xray/val', train=False)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    for i, (img, label) in enumerate(trainloader):
        if img.shape[1] == 3:
            plt.imshow(img[0,1], cmap='gray')
            plt.show()
        break