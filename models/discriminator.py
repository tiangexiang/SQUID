import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import numpy as np  


class SimpleDiscriminator(nn.Module):
    def __init__(self, size=4, inplace=True):
        super(SimpleDiscriminator, self).__init__()

        self.size = size

        keep_stats = True

        self.conv_model = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, 2, bias=True),
            #nn.BatchNorm2d(16, track_running_stats=keep_stats), # this maybe required
            nn.LeakyReLU(0.2, inplace=inplace),
            # group1
            nn.Conv2d(16, 32, 5, 2, 2, bias=True),
            nn.BatchNorm2d(32, track_running_stats=keep_stats),
            nn.LeakyReLU(0.2, inplace=inplace),
            # group3
            nn.Conv2d(32, 64, 5, 2, 2, bias=True),
            nn.BatchNorm2d(64, track_running_stats=keep_stats),
            nn.LeakyReLU(0.2, inplace=inplace),
            # group3
            nn.Conv2d(64, 128, 5, 2, 2, bias=True),
            nn.BatchNorm2d(128, track_running_stats=keep_stats),
            nn.LeakyReLU(0.2, inplace=inplace),
            # group4
            nn.Conv2d(128, 128, 5, 2, 2, bias=True),
            nn.BatchNorm2d(128, track_running_stats=keep_stats),
            nn.LeakyReLU(0.2, inplace=inplace),
        )

        self.regressor = nn.Linear(128 * size * size, 1)

    def forward(self, img):
        B = img.size(0)

        x = self.conv_model(img) # B, 128, W/16, H/16

        x = x.view(B, -1)
        x = self.regressor(x)
        return x
