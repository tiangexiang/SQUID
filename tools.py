import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt
import random

import copy
from dataloader.dataloader_digit import AbnomalMNIST, AbnomalMNISTTest
from dataloader.dataloader_zhang import Zhang
from dataloader.dataloader_chexpert import CheXpert
import shutil

import importlib
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='zhang_exp1')
    parser.add_argument('--config', type=str, default='zhang_best')
    args, unparsed = parser.parse_known_args()
    return args

def backup_files(args, model_file='squid'):
    # back up files
    shutil.copyfile('configs/'+args.config+'.py', os.path.join('checkpoints', args.exp, 'config.py'))
    shutil.copyfile('models/inpaint.py', os.path.join('checkpoints', args.exp, 'inpaint.py'))
    shutil.copyfile('models/memory.py', os.path.join('checkpoints', args.exp, 'memory.py'))
    shutil.copyfile('models/'+model_file+'.py', os.path.join('checkpoints', args.exp, model_file+'.py'))
    shutil.copyfile('models/discriminator.py', os.path.join('checkpoints', args.exp, 'discriminator.py'))
    shutil.copyfile('models/basic_modules.py', os.path.join('checkpoints', args.exp, 'basic_modules.py'))
    shutil.copyfile('main.py', os.path.join('checkpoints', args.exp, 'main.py'))
    shutil.copyfile('tools.py', os.path.join('checkpoints', args.exp, 'tools.py'))

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def build_disc(CONFIG):
    DISC = importlib.import_module('models.discriminator')

    if CONFIG.discriminator_type == 'basic':
        discriminator = DISC.SimpleDiscriminator(size=CONFIG.size).cuda()
        print('Basic discriminator created.')
    elif CONFIG.discriminator_type == 'tiny':
        discriminator = DISC.TinyDiscriminator(size=CONFIG.size).cuda()
        print('Tiny discriminator created.')
    elif CONFIG.discriminator_type == 'big':
        discriminator = DISC.BigDiscriminator(size=CONFIG.size).cuda()
        print('Big discriminator created.')
    elif CONFIG.discriminator_type == 'extratiny':
        discriminator = DISC.ExtraTinyDiscriminator(size=CONFIG.size).cuda()
        print('extratiny discriminator created.')
    return discriminator

def log(log_file, msg):
    log_file.write(msg+'\n')
    print(msg)

def log_loss(log_file, epoch, train_loss, val_loss):
    msg = 'Epoch: %d [TRAIN]' % epoch
    for k, v in train_loss.items():
        msg += ' %s: %.4f' % (k, v)
    msg += ' [VAL]'
    for k, v in val_loss.items():
        msg += ' %s: %.4f' % (k, v)
    log(log_file, msg)

def save_image(path, data):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    for idx, (img, target) in enumerate(data):
        plt.imsave(os.path.join(path, '%03d.jpg' % idx), img[0], cmap='gray')
        plt.imsave(os.path.join(path, 'input_%03d.jpg' % idx), target[0], cmap='gray')
