from argparse import ArgumentParser
from utils.utils import get_config
from dataloader import load_data, load_localization_data
from test_functions import detection_test, localization_test
from models.network import get_networks
from dataloader_zhang import Zhang
from dataloader_chexpert import ChexPert
import torch

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help="training configuration")


def main():
    args = parser.parse_args()
    config = get_config(args.config)
    vgg, model = get_networks(config, load_checkpoint=True)

        # zhang
    if config['dataset_name'] == 'zhang':
        testdataset = Zhang('zhanglab/test', train=False, 
                img_size=(128, 128), full=True)
        test_dataloader = torch.utils.data.DataLoader(testdataset, batch_size=32, shuffle=False, num_workers=0, drop_last=False)
    else:
        # chexpert
        testdataset = ChexPert('chexpert/our_test_256_'+'pa', train=False, 
                        img_size=(128, 128), full=True, data_type='pa')
        test_dataloader = torch.utils.data.DataLoader(testdataset, batch_size=32, shuffle=False, num_workers=0, drop_last=False)


    #_, test_dataloader = load_data(config)
    roc_auc = detection_test(model=model, vgg=vgg, test_dataloader=test_dataloader, config=config)
    last_checkpoint = config['last_checkpoint']
    #print("[TEST ]RocAUC after {} epoch:".format(last_checkpoint), roc_auc)


if __name__ == '__main__':
    main()
