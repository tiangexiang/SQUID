import math
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_window(x, window_size, stride=1, padding=0):
    """
    Args:
        x: (B, W, H, C)
        window_size (int): window size

    Returns:
        windows: (B*N,  ws**2, C)
    """
    x = x.permute(0, 3, 1, 2).contiguous()
    B, C, W, H = x.shape
    windows = F.unfold(x, window_size, padding=padding, stride=stride) # B, C*N, #of windows
    windows = windows.view(B, C, window_size**2, -1) #   B, C, ws**2, N
    windows = windows.permute(0, 3, 2, 1).contiguous().view(-1, window_size, window_size, C) # B*N, ws**2, C
    
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0., epsilon=1e-12):
    output = (F.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output

def binarize(integer, num_bits=8):   
    """Turn integer tensor to binary representation.        
    Args:           
    integer : torch.Tensor, tensor with integers           
    num_bits : Number of bits to specify the precision. Default: 8.       
    Returns:           
    Tensor: Binary tensor. Adds last dimension to original tensor for           
    bits.    
    """   
    dtype = integer.type()   
    exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)   
    exponent_bits = exponent_bits.repeat(integer.shape + (1,))   
    out = integer.unsqueeze(-1) / 2 ** exponent_bits   
    return (out - (out % 1)) % 2

def gumbel_softmax(att_weight, dim, k=1):
    y = F.softmax(att_weight, dim=dim)

    thres, _ = torch.topk(y, k, dim=1, sorted=True)
    thres = thres[:,[-1]] # N, 1

    y_hard = y.detach().clone()
    y_hard = hard_shrink_relu(y_hard, lambd=thres)  # [N,M]

    # normalize
    y_hard = F.normalize(y_hard, p=1, dim=1)  # [N,M]

    y_hard = (y_hard - y).detach() + y
    return y_hard


class MemoryQueue(nn.Module):
    def __init__(self, num_slots, slot_dim, shrink_thres=0.0025):
        super(MemoryQueue, self).__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        memMatrix = torch.zeros(num_slots, slot_dim)
        self.register_buffer('memMatrix', memMatrix)
        self.shrink_thres = shrink_thres
        self.ptr = nn.Parameter(torch.zeros(1,), requires_grad=False).long()

        if self.shrink_thres > 0. and type(self.shrink_thres) is int:
            print('[memory queue] Gumbel Shrinkage activated with threshold:', self.shrink_thres)
        elif self.shrink_thres > 0. and type(self.shrink_thres) is float:
            print('[memory queue] Hard Shrinkage activated with threshold:', self.shrink_thres)
        
        self.values = None

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.memMatrix.size(1))
        self.memMatrix.data.uniform_(-stdv, stdv)

    def enque(self, values):
        self.values = values.clone().detach()

    def update(self):
        #print('???')
        if self.values is None:
            return
    
        # values: B, C
        values = self.values

        # only support single gpu at this stage
        B, C = values.shape

        if self.ptr + B < self.num_slots:
            self.memMatrix[self.ptr:self.ptr+B] = values
        else:
            self.memMatrix[self.ptr:] = values[:self.num_slots - self.ptr]
            offset = (self.ptr + B) % self.num_slots
            self.memMatrix[:offset] = values[self.num_slots - self.ptr:]
        
        self.ptr = (self.ptr + B) % self.num_slots
        self.values = None
        del self.values

    def forward(self, x):
        """
        :param x: query features with size [N,C], where N is the number of query items,
                  C is same as dimension of memory slot

        :return: query output retrieved from memory, with the same size as x.
        """
        # dot product
        att_weight = F.linear(input=x, weight=self.memMatrix.detach())  # [N,C] by [M,C]^T --> [N,M]
        
        if self.shrink_thres > 0 and type(self.shrink_thres) is int:
            att_weight = gumbel_softmax(att_weight, dim=1, k=self.shrink_thres)
        elif self.shrink_thres > 0 and type(self.shrink_thres) is float:
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)  # [N,M]
            att_weight = F.normalize(att_weight, p=1, dim=1)
    
        out = F.linear(att_weight, self.memMatrix.permute(1, 0).detach())  # [N,M] by [M,C]  --> [N,C]
        return out


class Memory(nn.Module):
    def __init__(self, num_slots, slot_dim, shrink_thres=0.0025):
        super(Memory, self).__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        self.memMatrix = nn.Parameter(torch.empty(num_slots, slot_dim))  # M,C
        self.shrink_thres = shrink_thres

        if self.shrink_thres > 0. and type(self.shrink_thres) is int:
            print('[memory matrix] Gumbel Shrinkage activated with threshold:', self.shrink_thres)
        elif self.shrink_thres > 0. and type(self.shrink_thres) is float:
            print('[memory matrix] Hard Shrinkage activated with threshold:', self.shrink_thres)
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.memMatrix.size(1))
        self.memMatrix.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        :param x: query features with size [N,C], where N is the number of query items,
                  C is same as dimension of memory slot

        :return: query output retrieved from memory, with the same size as x.
        """
        # dot product
        att_weight = F.linear(input=x, weight=self.memMatrix)  # [N,C] by [M,C]^T --> [N,M]

        if self.shrink_thres > 0 and type(self.shrink_thres) is int:
            att_weight = gumbel_softmax(att_weight, dim=1, k=self.shrink_thres)
        elif self.shrink_thres > 0 and type(self.shrink_thres) is float:
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)  # [N,M]
            att_weight = F.normalize(att_weight, p=1, dim=1)

        out = F.linear(att_weight, self.memMatrix.permute(1, 0))  # [N,M] by [M,C]  --> [N,C]
        # sparse_loss = torch.mean(torch.sum(-att_weight * torch.log(att_weight + 1e-12), dim=1))
        return dict(out=out)


class MemoryMatrixBlock(nn.Module):
    '''
    Cross space-aware memory matrix block
    This performs the best, idk why :D
    '''
    def __init__(self, num_slots, slot_dim, num_memory=9, shrink_thres=0.0025, ratio=0.95):
        super(MemoryMatrixBlock, self).__init__()
        self.num_memory = num_memory
        self.memory = nn.ModuleList()
        self.mask_ratio = ratio
        for i in range(num_memory):
            self.memory.append(Memory(num_slots, slot_dim, shrink_thres, ))

    def forward(self, x):
        # x: b, c, w, h
        B, C, W, H = x.size()

        ox = x
        window_size = x.shape[2] // int(self.num_memory**0.5)
        x = make_window(x.permute(0, 2, 3, 1), window_size, stride=window_size, padding=0).view(B, self.num_memory, window_size**2, C) # B_, 3*3, C
        x = x.view(B, self.num_memory, -1)

        mem_styles = torch.zeros_like(x)
        for i in range(self.num_memory):
            mem_styles[:,i,:] = mem_styles[:,i,:] + self.memory[i](x[:,i,:])["out"]

        x = mem_styles.view(-1, window_size, window_size, C)
        x = window_reverse(x, window_size, W, H).permute(0, 3, 1, 2).contiguous()

        if self.training:
            mask = torch.ones(x.size(0), 1, x.size(-2), x.size(-1)).to(x.device) * self.mask_ratio
            mask = torch.bernoulli(mask).float()
            x = x * mask + ox * (1. - mask)

        return x


class MemoryMatrixBlockV2(nn.Module):
    '''
    Basic space-aware memory matrix block with bottleneck
    '''
    def __init__(self, in_channels, num_slots, slot_dim, num_memory=9, shrink_thres=0.0025, ratio=0.95):
        super(MemoryMatrixBlockV2, self).__init__()
        self.num_memory = num_memory
        self.memory = nn.ModuleList()
        self.mask_ratio = ratio
        for i in range(num_memory):
            self.memory.append(Memory(num_slots, slot_dim, shrink_thres, ))

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels//4, in_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        # x: b, c, w, h
        B, C, W, H = x.size()
        ox = x

        x = self.conv1(x)
        
        x = x.view(B//self.num_memory, self.num_memory, -1)
 
        mem_styles = torch.zeros_like(x)
        for i in range(self.num_memory):
            mem_styles[:,i,:] = mem_styles[:,i,:] + self.memory[i](x[:,i,:])["out"]
        x = mem_styles.view(B//self.num_memory, self.num_memory, -1, W, H)
        x = x.view(B, -1, W, H)
        
        x = F.relu(self.conv2(x), inplace=True)

        if self.training:
            mask = torch.ones(x.size(0), 1, x.size(-2), x.size(-1)).to(x.device) * self.mask_ratio
            mask = torch.bernoulli(mask).float()
            x = x * mask + ox * (1. - mask)
    
        return x

class MemoryMatrixBlockV3(nn.Module):
    '''
    Basic space-aware memory matrix block
    '''
    def __init__(self, in_channels, num_slots, slot_dim, num_memory=9, shrink_thres=0.0025, ratio=0.95):
        super(MemoryMatrixBlockV3, self).__init__()
        self.num_memory = num_memory
        self.memory = nn.ModuleList()
        self.mask_ratio = ratio
        for i in range(num_memory):
            self.memory.append(Memory(num_slots, slot_dim, shrink_thres, ))

    def forward(self, x):
        B, C, W, H = x.size()
        ox = x

        x = x.view(B//self.num_memory, self.num_memory, -1)

        mem_styles = torch.zeros_like(x)
        for i in range(self.num_memory):
            mem_styles[:,i,:] = self.memory[i](x[:,i,:])["out"]
        x = mem_styles.view(B//self.num_memory, self.num_memory, -1, W, H)
        x = x.view(B, -1, W, H)

        if self.training:
            mask = torch.ones(x.size(0), 1, x.size(-2), x.size(-1)).to(x.device) * self.mask_ratio
            mask = torch.bernoulli(mask).float()
            x = x * mask + ox * (1. - mask)
        return x


class MemoryMatrixBlockV4(nn.Module):
    '''
    None spatial-aware memory matrix block
    '''
    def __init__(self, num_slots, slot_dim, num_memory=9, shrink_thres=0.0025, ratio=0.95):
        super(MemoryMatrixBlockV4, self).__init__()
        self.num_memory = num_memory
        self.memory = nn.ModuleList()
        self.mask_ratio = ratio
        self.memory = Memory(num_slots, slot_dim, shrink_thres, )

    def forward(self, x):
        # x: b, c, w, h
        B, C, W, H = x.size()
        ox = x
        window_size = x.shape[2] // int(self.num_memory**0.5)
        x = make_window(x.permute(0, 2, 3, 1), window_size, stride=window_size, padding=0).view(B, self.num_memory, window_size**2, C) # B_, 3*3, C
        x = x.view(B, -1)

        mem_styles = self.memory(x)["out"]

        x = mem_styles.view(-1, window_size, window_size, C)
        x = window_reverse(x, window_size, W, H).permute(0, 3, 1, 2).contiguous()
        if self.training:
            mask = torch.ones(x.size(0), 1, x.size(-2), x.size(-1)).to(x.device) * self.mask_ratio
            mask = torch.bernoulli(mask).float()
            x = x * mask + ox * (1. - mask)
        return x

if __name__ == '__main__':
    q = MemoryQueue(10, 1, shrink_thres=5)

    for i in range(5):
        randinput = torch.randint(0,100, (3,1))
        q.enque(randinput)
        q.update()
        print('input is:',randinput)
        print('queue is:',q.memMatrix)
