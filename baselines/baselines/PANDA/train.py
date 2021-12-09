import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc, roc_curve, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
import torch.optim as optim
import argparse
from losses import CompactnessLoss, EWCLoss
import utils
from copy import deepcopy
from tqdm import tqdm
from dataloader_zhang import Zhang
from dataloader_chexpert import ChexPert

dataset = 'zhang' # 'chexpert'

def train_model(model, train_loader, test_loader, device, args, ewc_loss):
     #eval()
    auc, feature_space = get_score(model, device, train_loader, test_loader)
    print('Epoch: {}, AUROC is: {}'.format(0, auc))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005, momentum=0.9)
    center = torch.FloatTensor(feature_space).mean(dim=0)
    criterion = CompactnessLoss(center.to(device))
    best_auc = -1
    for epoch in range(args.epochs):
        running_loss = run_epoch(model, train_loader, optimizer, criterion, device, args.ewc, ewc_loss)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
        auc, feature_space = get_score(model, device, train_loader, test_loader)
        print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))
        if auc > best_auc:
            best_auc = auc
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
                   'checkpoints/model_'+dataset+'.pth')


def run_epoch(model, train_loader, optimizer, criterion, device, ewc, ewc_loss):
    running_loss = 0.0
    model.train()
    for i, (imgs, _) in enumerate(train_loader):

        images = imgs.to(device)

        optimizer.zero_grad()

        _, features = model(images)

        loss = criterion(features)

        if ewc:
            loss += ewc_loss(model)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / (i + 1)


def get_score(model, device, train_loader, test_loader):
    model.eval()
    train_feature_space = []
    with torch.no_grad():
        for (imgs, _) in tqdm(train_loader, desc='Train set feature extracting'):
            imgs = imgs.to(device)
            _, features = model(imgs)
            train_feature_space.append(features)
            #print(features.shape)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
    test_feature_space = []
    test_labels = []

    with torch.no_grad():
        for (imgs, label) in tqdm(test_loader, desc='Test set feature extracting'):
            imgs = imgs.to(device)
            _, features = model(imgs)
            test_feature_space.append(features)
            test_labels += list(label[:,0].cpu().numpy())
            #print(label.shape)
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        #test_labels = #test_loader.dataset.targets

    distances = utils.knn_score(train_feature_space, test_feature_space)
    test_labels = np.array(test_labels)
    #auc = roc_auc_score(test_labels, distances)

    scores = distances
    labels = test_labels
    #scores = np.array(scores)
    scores = -1 * scores
    fpr, tpr, thresholds = roc_curve(labels, scores)

    roc_auc = auc(fpr, tpr)
    roc_auc = round(roc_auc, 4)

    best_acc = 0.
    # labels = labels[:,0]
    print(labels.shape, scores.shape)
    
    for t in thresholds:

        prediction = np.zeros_like(scores)
        prediction[scores >= t] = 1

        # metrics
        f1 = f1_score(labels, prediction) * 100.
        acc = np.average(prediction == labels) * 100.
        #print(np.sum(prediction))
        recall = recall_score(labels, prediction) * 100.
        precision = precision_score(labels, prediction) * 100.
        #print(acc, f1)

        tn, fp, fn, tp = confusion_matrix(labels, prediction).ravel()
        specificity = (tn / (tn+fp)) * 100.
        
        if acc > best_acc:
            best_acc = acc
            results = dict(threshold=t, auc=roc_auc*100., acc=acc, f1=f1, recall=recall, precision=precision, specificity=specificity)

    msg = ''
    for k, v in results.items():
        msg += k + ': '
        msg += '%.2f ' % v
    print(msg)


    return roc_auc, train_feature_space

def main(args):
    print('Dataset: {}, Normal Label: {}, LR: {}'.format(args.dataset, args.label, args.lr))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = utils.get_resnet_model(resnet_type=args.resnet_type)
    model = model.to(device)

    ewc_loss = None

    # Freezing Pre-trained model for EWC
    if args.ewc:
        frozen_model = deepcopy(model).to(device)
        frozen_model.eval()
        utils.freeze_model(frozen_model)
        fisher = torch.load(args.diag_path)
        ewc_loss = EWCLoss(frozen_model, fisher)

    #utils.freeze_parameters(model)
    if dataset == 'zhang':
        #train_loader, test_loader = utils.get_loaders(dataset=args.dataset, label_class=args.label, batch_size=args.batch_size)

        traindataset = Zhang('zhanglab/train', train=True, img_size=(128, 128), enable_transform=True)
        valdataset = Zhang('zhanglab/val', train=False, img_size=(128, 128), full=True)

        train_loader = torch.utils.data.DataLoader(traindataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
        test_loader = torch.utils.data.DataLoader(valdataset, batch_size=32, shuffle=False, num_workers=0, drop_last=False)

    else:
        # chexpert
        traindataset = ChexPert('chexpert/train_256_'+'pa', 
                        train=True, img_size=(128, 128), enable_transform=True, data_type='pa')
        # testdataset = ChexPert('/media/administrator/1305D8BDB8D46DEE/jhu/CheXpert-v1.0-small/CheXpert-v1.0-small/our_test_256_'+'pa', train=False, 
        #                 img_size=(128, 128), full=True, data_type='pa')
        valdataset = ChexPert('chexpert/val_256_'+'pa', train=False, img_size=(128, 128), full=True, data_type='pa')
        train_loader = torch.utils.data.DataLoader(traindataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
        test_loader = torch.utils.data.DataLoader(valdataset, batch_size=32, shuffle=False, num_workers=0, drop_last=False)

    train_model(model, train_loader, test_loader, device, args, ewc_loss)


def test(args):
    print('Dataset: {}, Normal Label: {}, LR: {}'.format(args.dataset, args.label, args.lr))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = utils.get_resnet_model(resnet_type=args.resnet_type)
    model = model.to(device)

    if dataset == 'zhang':
        ckpt = torch.load('checkpoints/model_zhang.pth')
        model.load_state_dict(ckpt['state_dict'])
        train_loader, test_loader = utils.get_loaders(dataset=args.dataset, label_class=args.label, batch_size=args.batch_size)

        traindataset = Zhang('zhanglab/train', train=True, img_size=(128, 128), enable_transform=True)
        testdataset = Zhang('zhanglab/test', train=False, img_size=(128, 128), full=True)

        train_loader = torch.utils.data.DataLoader(traindataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
        test_loader = torch.utils.data.DataLoader(testdataset, batch_size=32, shuffle=False, num_workers=0, drop_last=False)

    else:
        # chexpert
        ckpt = torch.load('checkpoints/model_chexpert.pth')
        model.load_state_dict(ckpt['state_dict'])
        traindataset = ChexPert('chexpert/train_256_'+'pa', 
                        train=True, img_size=(128, 128), enable_transform=True, data_type='pa')
        testdataset = ChexPert('chexpert/our_test_256_'+'pa', train=False, 
                        img_size=(128, 128), full=True, data_type='pa')
 
        train_loader = torch.utils.data.DataLoader(traindataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
        test_loader = torch.utils.data.DataLoader(testdataset, batch_size=32, shuffle=False, num_workers=0, drop_last=False)

    get_score(model, device, train_loader, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--diag_path', default='./data/fisher_diagonal.pth', help='fim diagonal path')
    parser.add_argument('--ewc', action='store_true', help='Train with EWC')
    parser.add_argument('--epochs', default=50, type=int, metavar='epochs', help='number of epochs')
    parser.add_argument('--label', default=0, type=int, help='The normal class')
    parser.add_argument('--lr', type=float, default=1e-4, help='The initial learning rate.')
    parser.add_argument('--resnet_type', default=50, type=int, help='which resnet to use')
    parser.add_argument('--batch_size', default=32, type=int)

    args = parser.parse_args()

    main(args)

    #test(args)
