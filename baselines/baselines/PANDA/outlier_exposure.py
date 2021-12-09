import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import argparse
import utils

def train_model(model, train_loader, outliers_loader, test_loader, device, epochs, lr):
    model.eval()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    bce = torch.nn.BCELoss()
    for epoch in range(epochs):
        running_loss = run_epoch(model, train_loader, outliers_loader, optimizer, bce, device)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
        auc = get_score(model, device, test_loader)
        print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))


def run_epoch(model, train_loader, outliers_loader, optimizer, bce, device):
    running_loss = 0.0
    for i, (imgs, _) in enumerate(train_loader):

        imgs = imgs.to(device)

        out_imgs, _ = next(iter(outliers_loader))

        outlier_im = out_imgs.to(device)

        optimizer.zero_grad()

        pred, _ = model(imgs)
        outlier_pred, _ = model(outlier_im)

        batch_1 = pred.size()[0]
        batch_2 = outlier_pred.size()[0]

        labels = torch.zeros(size=(batch_1 + batch_2,), device=device)
        labels[batch_1:] = torch.ones(size=(batch_2,))

        loss = bce(torch.sigmoid(torch.cat([pred, outlier_pred])), labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / (i + 1)



def get_score(model, device, test_loader):
    model.eval()
    anom_labels = []
    predictions = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.numpy()
            pred, _ = model(imgs)
            pred = torch.sigmoid(pred)
            batch_size = imgs.shape[0]
            for j in range(batch_size):
                predictions.append(pred[j].detach().cpu().numpy())
                anom_labels.append(labels[j])

    test_set_predictions = np.array(predictions)
    test_labels = np.array(anom_labels)

    auc = roc_auc_score(test_labels, test_set_predictions)

    return auc

def main(args):
    print('Dataset: {}, Label: {}, LR: {}'.format(args.dataset, args.label, args.lr))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = utils.get_resnet_model(resnet_type=args.resnet_type)

    # Change last layer
    model.fc = torch.nn.Linear(args.latent_dim_size, 1)

    model = model.to(device)
    utils.freeze_parameters(model, train_fc=True)

    train_loader, test_loader = utils.get_loaders(dataset=args.dataset, label_class=args.label, batch_size=args.batch_size)
    outliers_loader = utils.get_outliers_loader(args.batch_size)

    train_model(model, train_loader, outliers_loader, test_loader, device, args.epochs, args.lr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--epochs', default=50, type=int, metavar='epochs', help='number of epochs')
    parser.add_argument('--label', default=0, type=int, help='The normal class')
    parser.add_argument('--lr', type=float, default=1e-1, help='The initial learning rate.')
    parser.add_argument('--resnet_type', default=152, type=int, help='which resnet to use')
    parser.add_argument('--latent_dim_size', default=2048, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    args = parser.parse_args()

    main(args)
