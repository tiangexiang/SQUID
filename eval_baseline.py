import torch
torch.set_printoptions(10)

import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import numpy as np
import torch.optim as optim
import os
import shutil
from matplotlib import pyplot as plt

from models.baseline import AE

import random
import argparse

import importlib
from sklearn.metrics import auc, roc_curve, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

from tools import parse_args, build_disc, log, log_loss, save_image


args = parse_args()

if not os.path.exists(os.path.join('checkpoints', args.exp)):
    print('exp folder cannot be found!')
    exit()

if not os.path.isfile(os.path.join('checkpoints', args.exp, 'config.py')):
    print('config file cannot be found!')
    exit()

# load config file from exp folder
CONFIG = importlib.import_module('checkpoints.'+args.exp+'.config').Config()

save_path = os.path.join('checkpoints', args.exp, 'test_images')

# log
log_file = open(os.path.join('checkpoints', args.exp, 'eval_log.txt'), 'w')

# build main model from exp folder
MODULE = importlib.import_module('checkpoints.'+args.exp+'.baseline')
model = MODULE.AE(1, 32, CONFIG.shrink_thres, num_slots=CONFIG.num_slots, num_patch=CONFIG.num_patch, level=CONFIG.level, 
            ratio=CONFIG.mask_ratio, initial_combine=CONFIG.initial_combine, drop=CONFIG.drop,
            dist=CONFIG.dist, memory_channel=CONFIG.memory_channel, mem_num_slots=CONFIG.mem_num_slots,
            ).cuda()

print('Loading AE...')
ckpt = torch.load(os.path.join('checkpoints',args.exp,'model.pth'))
model.load_state_dict(ckpt)
print('AE loaded!')


def evaluation():
    reconstructed, inputs, results, val_loss = test(CONFIG.test_loader)

    msg = '[TEST metrics] '
    for k, v in results.items():
        msg += k + ': '
        msg += '%.2f ' % v
    log(log_file, msg)

    save_image(os.path.join(save_path, 'test'), zip(reconstructed, inputs))

def test(dataloader):
    model.eval()
    tot_loss = {'recon_l1': 0.}
    # for reconstructed img
    reconstructed = []
    # for input img
    inputs = []
    # for anomaly score
    scores = []
    # for gt labels
    labels = []

    count = 0
    for i, (img, label) in enumerate(dataloader):
        count += img.shape[0]
        img = img.to(CONFIG.device)
        label = label.to(CONFIG.device)

        out = model(img)

        score = torch.sqrt((out['recon'] - img)**2)
        score = score.view(score.shape[0], -1)
        score = torch.mean(score, dim=-1)
        scores += list(score.detach().cpu().numpy())

        tot_loss['recon_l1'] += torch.mean(torch.abs(out['recon'] - img)).item()

        labels += list(label.detach().cpu().numpy())
        reconstructed += list(out['recon'].detach().cpu().numpy())
        inputs += list(img.detach().cpu().numpy())
    
    scores = np.array(scores)
    labels = np.array(labels)
    labels = labels[:,0]

    #print(labels.shape, scores.shape)

    fpr, tpr, thres = roc_curve(labels, scores)
    auc_score = auc(fpr, tpr) * 100.

    best_acc = 0.

    for t in thres:
        prediction = np.zeros_like(scores)
        prediction[scores >= t] = 1

        # metrics
        f1 = f1_score(labels, prediction) * 100.
        acc = np.average(prediction.astype(np.bool8) == labels.astype(np.bool8)) * 100.
        recall = recall_score(labels, prediction) * 100.
        precision = precision_score(labels, prediction) * 100.
        tn, fp, fn, tp = confusion_matrix(labels, prediction).ravel()
        specificity = (tn / (tn+fp)) * 100.

        if acc > best_acc:
            best_acc = acc
            results = dict(threshold=t, auc=auc_score, acc=acc, f1=f1, recall=recall, precision=precision, specificity=specificity)
    
    tot_loss['recon_l1'] = tot_loss['recon_l1'] / count
    return reconstructed, inputs, results, tot_loss

if __name__ == '__main__':
    evaluation()
