import math
import sys
import os
import importlib

from models.basic_modules import *
import models.memory as Memory
from models.inpaint import InpaintBlock


class AE(nn.Module):
    def __init__(self, num_in_ch, features_root, shrink_thres, 
                 num_slots=200, num_patch=2, level=4, ratio=0.95, 
                 drop=0.0, memory_channel=2048, dist=False, initial_combine=None, mem_num_slots=200,
                 ops=['concat', 'concat', 'none', 'none'], decoder_memory=None):
        super(AE, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_slots = num_slots
        self.shrink_thres = shrink_thres
        self.initial_combine = initial_combine
        self.level = level
        self.num_patch = num_patch
        self.drop = drop
        print('SQUID ops:', ops)
        
        assert len(ops) == level # make sure there is an op for every decoder level

        self.filter_list = [features_root, features_root*2, features_root*4, features_root*8, features_root*16, features_root*16]

        self.in_conv = inconv(num_in_ch, features_root)
        
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        for i in range(level):
            self.down_blocks.append(down(self.filter_list[i], self.filter_list[i+1], use_se=False))
            if ops[i] == 'concat':
                filter = self.filter_list[level-i] + self.filter_list[level-i-1]#//2
            else:
                filter = self.filter_list[level-i]
            self.up_blocks.append(up(filter, self.filter_list[level-1-i], op=ops[i], use_se=False))

        self.inpaint_block = InpaintBlock(self.filter_list[level], num_slots, num_memory=self.num_patch**2, memory_channel=memory_channel, shrink_thres=shrink_thres, ratio=ratio, drop=drop)
        
        assert decoder_memory is not None # decoder memory should NOT be none in all cases

        self.memory_blocks = nn.ModuleList()
        for i, config in enumerate(decoder_memory):
            if config is None:
                self.memory_blocks.append(nn.Identity())
            else:
                self.memory_blocks.append(getattr(Memory, config['type'])(mem_num_slots, 
                                                                          self.filter_list[i] * config['multiplier'], 
                                                                          num_memory=config['num_memory'],
                                                                          shrink_thres=shrink_thres))

        self.out_conv = outconv(features_root, num_in_ch)

        self.mse_loss = nn.MSELoss()

        self.dist = dist
        if dist:
            self.teacher_ups = nn.ModuleList()
            for i in range(level):
                if ops[i] == 'concat':
                    filter = self.filter_list[level-i] + self.filter_list[level-i-1]
                else:
                    filter = self.filter_list[level-i]
                self.teacher_ups.append(up(filter, self.filter_list[level-1-i], op=ops[i], use_se=False))
            self.teacher_out = outconv(features_root, num_in_ch)

    def forward(self, x):
        """
        :param x: size [bs,C,H,W]
        :return:
        """
        bs, C, W, H = x.size()
        assert W % self.num_patch == 0 or  H % self.num_patch == 0

        # segment patches
        x = make_window(x.permute(0, 2, 3, 1).contiguous(), W//self.num_patch, H//self.num_patch, 0).permute(0, 3, 1, 2) # B * 9, C, ws, ws
        num_windows = x.size(0) // bs

        x = self.in_conv(x)

        skips = []
        # encoding
        for i in range(self.level):
            B_, c, w, h = x.size()
            if i < self.initial_combine:
                sc = window_reverse(x.permute(0, 2, 3, 1).contiguous(), w, w * self.num_patch, w * self.num_patch).permute(0, 3, 1, 2)
                skips.append(sc)
            else:
                skips.append(x)

            x = self.down_blocks[i](x)

        B_, c, w, h = x.shape

        # this is useless currently, but could be useful in the future
        embedding = window_reverse(x.permute(0, 2, 3, 1).contiguous(), w, w * self.num_patch, w * self.num_patch).permute(0, 3, 1, 2).contiguous()
        
        t_x = x.clone().detach()
  
        x = self.inpaint_block(x, bs, num_windows, add_condition=True)

        self_dist_loss = []
        # decoding
        for i in range(self.level):
            #print(x.shape)
            # combine patches?
            if self.initial_combine is not None and self.initial_combine == (self.level - i):
                B_, c, w, h = x.shape
                x = window_reverse(x.permute(0, 2, 3, 1).contiguous(), w, w * self.num_patch, w * self.num_patch).permute(0, 3, 1, 2)
                #print(x.shape,'??')
                t_x = window_reverse(t_x.permute(0, 2, 3, 1).contiguous(), w, w * self.num_patch, w * self.num_patch).permute(0, 3, 1, 2)
    
            x = self.up_blocks[i](x, skips[-1-i])
  
            # additional decoder memory matrix
            x = self.memory_blocks[-1-i](x)

            if self.dist:
                t_x = self.teacher_ups[i](t_x, skips[-1-i].detach().clone())

                # do we need sg here? maybe not
                self_dist_loss.append(self.mse_loss(x, t_x))

        # forward teacher decoder
        if self.dist:
            self_dist_loss = torch.sum(torch.stack(self_dist_loss))
            t_x = self.teacher_out(t_x)
            t_x = torch.sigmoid(t_x)
            B_, c, w, h = t_x.shape
            if self.initial_combine is None:
                t_x = window_reverse(t_x.permute(0, 2, 3, 1).contiguous(), w, w * self.num_patch, w * self.num_patch).permute(0, 3, 1, 2)

        x = self.out_conv(x)
        x = torch.sigmoid(x)

        B_, c, w, h = x.shape

        if self.initial_combine is None:
            whole_recon = window_reverse(x.permute(0, 2, 3, 1).contiguous(), w, w * self.num_patch, w * self.num_patch).permute(0, 3, 1, 2)
        else:
            whole_recon = x
            x = make_window(x.permute(0, 2, 3, 1).contiguous(), W//self.num_patch, H//self.num_patch, 0).permute(0, 3, 1, 2)

        outs = dict(recon=whole_recon, patch=x, embedding=embedding, teacher_recon=t_x, dist_loss=self_dist_loss)
        return outs
