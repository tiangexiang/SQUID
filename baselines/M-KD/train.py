from test import *
from utils.utils import *
from dataloader import *
from dataloader_zhang import Zhang
from pathlib import Path
from torch.autograd import Variable
import pickle
from test_functions import detection_test
from dataloader_chexpert import ChexPert
from loss_functions import *

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help="training configuration")


def train(config):
    direction_loss_only = config["direction_loss_only"]
    normal_class = config["normal_class"]
    learning_rate = float(config['learning_rate'])
    num_epochs = config["num_epochs"]
    lamda = config['lamda']
    continue_train = config['continue_train']
    last_checkpoint = config['last_checkpoint']

    checkpoint_path = "./outputs/{}/{}/checkpoints/".format(config['experiment_name'], config['dataset_name'])

    # create directory
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    #train_dataloader, test_dataloader = load_data(config)

    if config['dataset_name'] == 'zhang':
        # zhang
        traindataset = Zhang('zhanglab/train', train=True, img_size=(128, 128), enable_transform=True)
        valdataset = Zhang('zhanglab/val', train=False, img_size=(128, 128), full=True)
        #testdataset = Zhang('zhanglab/test', train=False, img_size=(128, 128), full=True)

        train_dataloader = torch.utils.data.DataLoader(traindataset, batch_size=32, shuffle=True, num_workers=0, drop_last=False)
        #val_dataloader = torch.utils.data.DataLoader(valdataset, batch_size=32, shuffle=False, num_workers=0, drop_last=False)
        test_dataloader = torch.utils.data.DataLoader(valdataset, batch_size=32, shuffle=False, num_workers=0, drop_last=False)
    else:
        traindataset = ChexPert('chexpert/train_256_'+'pa', 
                        train=True, img_size=(128, 128), enable_transform=True, data_type='pa')
        # testdataset = ChexPert('chexpert/our_test_256_'+'pa', train=False, 
        #                 img_size=(128, 128), full=True, data_type='pa')
        valdataset = ChexPert('chexpert/val_256_'+'pa', train=False, 
                        img_size=(128, 128), full=True, data_type='pa')
        train_dataloader = torch.utils.data.DataLoader(traindataset, batch_size=32, shuffle=True, num_workers=0, drop_last=False)
        test_dataloader = torch.utils.data.DataLoader(valdataset, batch_size=32, shuffle=False, num_workers=0, drop_last=False)
        #testloader = torch.utils.data.DataLoader(testdataset, batch_size=opt.batchsize, shuffle=False, num_workers=0, drop_last=False)


    
    
    if continue_train:
        vgg, model = get_networks(config, load_checkpoint=True)
    else:
        vgg, model = get_networks(config)

    # Criteria And Optimizers
    if direction_loss_only:
        criterion = DirectionOnlyLoss()
    else:
        criterion = MseDirectionLoss(lamda)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if continue_train:
        optimizer.load_state_dict(
            torch.load('{}Opt_{}_epoch_{}.pth'.format(checkpoint_path, normal_class, last_checkpoint)))

    losses = []
    roc_aucs = []
    if continue_train:
        with open('{}Auc_{}_epoch_{}.pickle'.format(checkpoint_path, normal_class, last_checkpoint), 'rb') as f:
            roc_aucs = pickle.load(f)
    best_auc = -1
    for epoch in range(num_epochs + 1):
        model.train()
        epoch_loss = 0
        for data in train_dataloader:
            X = data[0]
            if X.shape[1] == 1:
                X = X.repeat(1, 3, 1, 1)
            X = Variable(X).cuda()

            output_pred = model.forward(X)
            output_real = vgg(X)

            total_loss = criterion(output_pred, output_real)

            # Add loss to the list
            epoch_loss += total_loss.item()
            losses.append(total_loss.item())

            # Clear the previous gradients
            optimizer.zero_grad()
            # Compute gradients
            total_loss.backward()
            # Adjust weights
            optimizer.step()

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        #if epoch % 10 == 0:
        roc_auc = detection_test(model, vgg, test_dataloader, config)
        roc_aucs.append(roc_auc)
        print("VAL RocAUC at epoch {}:".format(epoch), roc_auc)

        if roc_auc >= best_auc:
            best_auc = roc_auc
            torch.save(model.state_dict(),
                       '{}Cloner_{}.pth'.format(checkpoint_path, normal_class))
            torch.save(optimizer.state_dict(),
                       '{}Opt_{}.pth'.format(checkpoint_path, normal_class,))
            # with open('{}Auc_{}_epoch_{}.pickle'.format(checkpoint_path, normal_class, epoch),
            #           'wb') as f:
            #     pickle.dump(roc_aucs, f)


def main():
    args = parser.parse_args()
    config = get_config(args.config)
    train(config)


if __name__ == '__main__':
    main()
