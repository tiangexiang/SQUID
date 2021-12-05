import math
import sys
import os
import torch
from models.basic_modules import *
from models.memory import Memory


class CAE(nn.Module):
    def __init__(self, num_in_ch, features_root,
                 shrink_thres, num_slots=1000, num_patch=3, 
                 level=3, ratio=0.8, drop=0.5, memory_channel=2048,
                 dist=False, initial_combine=None, mem_num_slots=500):
        super(CAE, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_slots = num_slots
        self.shrink_thres = shrink_thres
        self.initial_combine = initial_combine
        self.level = level
        self.num_patch = num_patch
        self.drop = drop

        self.filter_list = [features_root, features_root*2, features_root*4, features_root*8, features_root*16, features_root*16]

        self.in_conv = inconv(num_in_ch, features_root)

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        ops = ['none', 'none', 'none', 'none', 'none']
        for i in range(level):
            self.down_blocks.append(down(self.filter_list[i], self.filter_list[i+1], use_se=False))
            if ops[i] == 'concat':
                filter = self.filter_list[level-i] + self.filter_list[level-i]//2
            else:
                filter = self.filter_list[level-i]
            self.up_blocks.append(up(filter, self.filter_list[level-1-i], op=ops[i], use_se=False))
        
        self.memory = Memory(num_slots, 18432, shrink_thres=0)

        self.out_conv = outconv(features_root, num_in_ch)

        self.mse_loss = nn.MSELoss()

    def forward(self, x):
        """
        :param x: size [bs,C,H,W]
        :return:
        """

        x = self.in_conv(x)

        shortcut_size = []
        for i in range(self.level):
            B_, c, w, h = x.size()
            shortcut_size.append(x)
            
            x = self.down_blocks[i](x)

        B_, c, w, h = x.shape

        x = x.view(B_, -1)
        x = self.memory(x)['out']
        x = x.view(B_, c, w, h)

        for i in range(self.level):
            x = self.up_blocks[i](x, shortcut_size[-1-i])
 
        x = self.out_conv(x) # B, 1, 84, 84
        x = torch.sigmoid(x)

        outs = dict(recon=x, patch=x, embedding=x, teacher_recon=torch.zeros((1,)).to(x.device), dist_loss=torch.zeros((1,)).to(x.device))
        return outs
