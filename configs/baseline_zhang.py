import sys

sys.path.insert(0, '..')

from dataloader.dataloader_zhang import *
from configs.base import BaseConfig

class Config(BaseConfig):
    def __init__(self):
        super(Config, self).__init__()

        #---------------------
        # Training Parameters
        #---------------------
        self.print_freq = 10
        self.device = 'cuda:0'
        self.epochs = 200
        self.lr = 1e-4 # learning rate
        self.batch_size = 16
        self.test_batch_size = 2
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
        self.scheduler_args = dict(T_max=300, eta_min=self.lr*0.1)

        # GAN # this WON'T be used
        self.gan_lr = 1e-4
        self.discriminator_type = 'basic'
        self.enbale_gan = 300 # no gan enabled!
        self.lambda_gp = 10.
        self.size = 4
        self.n_critic = 2
        self.sample_interval = 1000
        self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR
        self.scheduler_args_d = dict(T_max=200, eta_min=self.lr*0.2)

        # model
        self.num_patch = 1 #4
        self.level = 5 # there is no space dividing!
        self.shrink_thres = 0.0015
        self.initial_combine = 2 # this WON'T be used
        self.drop = 0.
        self.dist = True # this WON'T be used
        self.num_slots = 2000 # as in their paper
        self.mem_num_slots = 2000
        self.memory_channel = 2048
        self.img_size = 128

        # loss weight
        self.t_w = 0.01
        self.recon_w = 10.
        self.dist_w = 0.01
        self.g_w = 0.005
        self.d_w = 0.005


        self.train_dataset = Zhang(self.data_root+'/ZhangLabData/CellData/chest_xray/train', train=True, img_size=(self.img_size, self.img_size))
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=False)

        self.val_dataset = Zhang(self.data_root+'/ZhangLabData/CellData/chest_xray/val', train=False, img_size=(self.img_size, self.img_size), full=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

        self.test_dataset = Zhang(self.data_root+'/ZhangLabData/CellData/chest_xray/test', train=False, img_size=(self.img_size, self.img_size), full=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
