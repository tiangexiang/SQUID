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
from scipy.special import expit
from sklearn.metrics import auc, roc_curve, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix


class GanAlert(object):
    def __init__(self, discriminator, args, CONFIG=None, generator=None):
        self.args = args
        self.scores = []
        self.labels = []

        # for vis
        self.imgs = []
        self.targets = []

        self.discriminator = discriminator
        self.generator = generator

        self.CONFIG = CONFIG

        self.early_stop = CONFIG.early_stop if CONFIG is not None else 200

        # training set with batch size 1
        self.train_loader = torch.utils.data.DataLoader(CONFIG.train_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)


    def collect(self, dataloader):
        assert self.generator is not None
        self.generator.eval()

        all_disc = []

        for i, (img, label) in enumerate(dataloader):
            B = img.shape[0]

            img = img.to(self.CONFIG.device)
            label = label.to(self.CONFIG.device)

            out = self.generator(img)

            fake_v = self.discriminator(out['recon'])
            disc = list(fake_v.detach().cpu().numpy())
            all_disc += disc

            if i >= self.early_stop:
                break

        # calculate stats
        return np.mean(all_disc), np.std(all_disc)

    def evaluate(self, scores, labels, collect=True):

        # calculate mean/std on training set?
        if collect:
            mean, std = self.collect(self.train_loader)
        else:
            mean = 0.
            std = 1.
        
        results = self.alert(scores, labels, mean, std, print_info=False)

        return results

    def alert(self, scores, labels, mean=0., std=1., print_info=True):
        scores = np.array(scores)
        labels = np.array(labels)
        
        scores = (scores - mean) / std

        scores = 1. - expit(scores) # 1 is anomaly!!
 
        best_acc = -1
        best_t = 0

        fpr, tpr, thres = roc_curve(labels, scores)

        auc_score = auc(fpr, tpr) * 100.

        for t in thres:
            prediction = np.zeros_like(scores)
            prediction[scores >= t] = 1

            # metrics
            f1 = f1_score(labels, prediction) * 100.
            acc = np.average(prediction == labels) * 100.
            recall = recall_score(labels, prediction) * 100.
            precision = precision_score(labels, prediction, labels=np.unique(prediction)) * 100.
            tn, fp, fn, tp = confusion_matrix(labels, prediction).ravel()
            specificity = (tn / (tn+fp)) * 100.

            if acc > best_acc:
                best_t = t
                best_acc = acc
                results = dict(threshold=t, auc=auc_score, acc=acc, f1=f1, recall=recall, precision=precision, specificity=specificity)

            if print_info:
                print('threshold: %.2f, auc: %.2f, acc: %.2f, f1: %.2f, recall(sens): %.2f, prec: %.2f, spec: %.2f' % (t, auc_score, acc, f1, recall, precision, specificity))

        if print_info:
            print('[BEST] threshold: %.2f, auc: %.2f, acc: %.2f, f1: %.2f, recall(sens): %.2f, prec: %.2f, spec: %.2f' % (results['threshold'], results['auc'], results['acc'], results['f1'], results['recall'], results['precision'], results['specificity']))

        return results
