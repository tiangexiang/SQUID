import torch
import torch.nn.functional as F
import torch
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
from matplotlib import pyplot as plt
import random

class DigitAnatomy(torch.utils.data.Dataset):
    def __init__(self, root, train=True, length=100, img_size=32):
        self.length = length
        self.data = {}
        self.data_num = {}
        self.img_size = img_size

        # load mnist
        for i in range(10):
            self.data[i] = np.load(os.path.join(root, str(i)+'.npy'))
            self.data_num[i] = self.data[i].shape[0]

        # random scale
        scaler =transforms.RandomResizedCrop(size=(self.data[0].shape[1], self.data[0].shape[2] ),
                                             scale =(0.6, 1.3), ratio=(1.0, 1.0))
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            scaler,
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])

    def make_normal_img(self):
        rows = []
        row = []
        for i in range(1, 10):
            data = self.data[i][np.random.randint(0, self.data_num[i])]
            # random scale
            data = self.transforms(data)
            row.append(data)
            if i % 3 == 0:
                row = torch.cat(row, dim=2)
                rows.append(row)
                row = []
        img = torch.cat(rows, dim=1)
        return img

    def __getitem__(self, index):
        img = self.make_normal_img()

        return img,  torch.zeros((1,))

    def __len__(self):
        return self.length

class DigitAnatomyTest(torch.utils.data.Dataset):
    def __init__(self, root, img_size=32):
        self.data = {}
        self.img_size = img_size
        
        # load data
        items = os.listdir(os.path.join(root, 'abnormal'))
        self.abnormal_data = [plt.imread(os.path.join(root, 'abnormal', item))[:,:,0] for item in items]
        self.abnormal_label = [1 for _ in range(len(self.abnormal_data))]
        items = os.listdir(os.path.join(root, 'normal'))
        self.normal_data = [plt.imread(os.path.join(root, 'normal', item))[:,:,0] for item in items]
        self.normal_label = [0 for _ in range(len(self.normal_data))]
    
        self.data = self.abnormal_data + self.normal_data
        self.labels = self.abnormal_label + self.normal_label

        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size * 3),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img = self.data[index]
        img = self.transforms(img)

        label = self.labels[index]

        return img,  torch.zeros((1,)) + label

    def __len__(self):
        return len(self.data)
