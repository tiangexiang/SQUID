import torch
torch.set_printoptions(10)

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os
import shutil
from matplotlib import pyplot as plt

from models.baseline import AE

import random
import importlib
import copy
from tqdm import tqdm
from sklearn.metrics import auc, roc_curve, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

from tools import parse_args, build_disc, log, log_loss, save_image, backup_files
from alert import GanAlert


args = parse_args()

CONFIG = importlib.import_module('configs.'+args.config).Config()

if not os.path.exists(os.path.join('checkpoints', args.exp)):
    os.mkdir(os.path.join('checkpoints', args.exp))

if not os.path.exists(os.path.join('checkpoints', args.exp, 'test_images')):
    os.mkdir(os.path.join('checkpoints', args.exp, 'test_images'))

save_path = os.path.join('checkpoints', args.exp, 'test_images')

# log
log_file = open(os.path.join('checkpoints', args.exp, 'log.txt'), 'w')

# backup files
backup_files(args, model_file='baseline')

# main model
model = AE(1, 32, CONFIG.shrink_thres, num_slots=CONFIG.num_slots, num_patch=CONFIG.num_patch, level=CONFIG.level, 
            ratio=CONFIG.mask_ratio, initial_combine=CONFIG.initial_combine, drop=CONFIG.drop,
            dist=CONFIG.dist, memory_channel=CONFIG.memory_channel, mem_num_slots=CONFIG.mem_num_slots).cuda()
opt = CONFIG.opt(model.parameters(), lr=CONFIG.lr, eps=1e-7, betas=(0.5, 0.999), weight_decay=0.00001)
scheduler = CONFIG.scheduler(opt, **CONFIG.scheduler_args)

ce = nn.BCEWithLogitsLoss().cuda()
contrastive_criterion = torch.nn.CrossEntropyLoss().cuda()
recon_criterion = torch.nn.MSELoss(reduction='mean').cuda()

def main():

    best_auc = -1

    for epoch in range(CONFIG.epochs):

        train_loss = train(CONFIG.train_loader, epoch)
        
        reconstructed, inputs, results, val_loss = mse_val(CONFIG.val_loader, epoch)

        log_loss(log_file, epoch, train_loss, val_loss)

        # do we need scheduler for discriminator?
        scheduler.step()      
        
        # log metrics
        msg = '[VAL metrics] '
        for k, v in results.items():
            msg += k + ': '
            msg += '%.2f ' % v
        log(log_file, msg)

        # save best model
        if results['auc'] > best_auc - 0.5: # a little bit tolerance
            if results['auc'] > best_auc:
                best_auc = results['auc']
            save_image(os.path.join(save_path, 'best'), zip(reconstructed, inputs))
            torch.save(model.state_dict(), os.path.join('checkpoints',args.exp,'model.pth'))
            log(log_file, 'save model!')

        # save last 10 epochs generated imgs for debugging
        if epoch >= CONFIG.epochs - 10:
            save_image(os.path.join(save_path, 'epoch_'+str(epoch)), zip(reconstructed, inputs))

    log_file.close()

def train(dataloader, epoch):
    model.train()
    batches_done = 0
    tot_loss = {'recon_loss': 0.}
    
    # clip dataloader
    if CONFIG.limit is None:
        limit = len(dataloader) - len(dataloader) % CONFIG.n_critic
    else:
        limit = CONFIG.limit

    for i, (img, label) in enumerate(tqdm(dataloader, disable=CONFIG.disable_tqdm)):
        if i > limit:
            break
        batches_done += 1

        img = img.to(CONFIG.device)
        label = label.to(CONFIG.device)
        
        opt.zero_grad()
        
        out = model(img)

        loss_all = CONFIG.recon_w * recon_criterion(out["recon"], img)
        tot_loss['recon_loss'] += loss_all.item()

        loss_all.backward()
        opt.step()

    # avg loss
    for k, v in tot_loss.items():
        tot_loss[k] /= batches_done

    return tot_loss

def mse_val(dataloader, epoch):
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

        opt.zero_grad()

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
    main()