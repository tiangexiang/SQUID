import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torchvision.datasets import ImageFolder
from PIL import Image


def load_data(config):
    normal_class = config['normal_class']
    batch_size = config['batch_size']

    if config['dataset_name'] in ['cifar10']:
        img_transform = transforms.Compose([
            transforms.Resize((256, 256), Image.ANTIALIAS),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        os.makedirs("./Dataset/CIFAR10/train", exist_ok=True)
        dataset = CIFAR10('./Dataset/CIFAR10/train', train=True, download=True, transform=img_transform)
        print("Cifar10 DataLoader Called...")
        print("All Train Data: ", dataset.data.shape)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]
        print("Normal Train Data: ", dataset.data.shape)

        os.makedirs("./Dataset/CIFAR10/test", exist_ok=True)
        test_set = CIFAR10("./Dataset/CIFAR10/test", train=False, download=True, transform=img_transform)
        print("Test Train Data:", test_set.data.shape)

    elif config['dataset_name'] in ['mnist']:
        img_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        os.makedirs("./Dataset/MNIST/train", exist_ok=True)
        dataset = MNIST('./Dataset/MNIST/train', train=True, download=True, transform=img_transform)
        print("MNIST DataLoader Called...")
        print("All Train Data: ", dataset.data.shape)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]
        print("Normal Train Data: ", dataset.data.shape)

        os.makedirs("./Dataset/MNIST/test", exist_ok=True)
        test_set = MNIST("./Dataset/MNIST/test", train=False, download=True, transform=img_transform)
        print("Test Train Data:", test_set.data.shape)

    elif config['dataset_name'] in ['fashionmnist']:
        img_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        os.makedirs("./Dataset/FashionMNIST/train", exist_ok=True)
        dataset = FashionMNIST('./Dataset/FashionMNIST/train', train=True, download=True, transform=img_transform)
        print("FashionMNIST DataLoader Called...")
        print("All Train Data: ", dataset.data.shape)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]
        print("Normal Train Data: ", dataset.data.shape)

        os.makedirs("./Dataset/FashionMNIST/test", exist_ok=True)
        test_set = FashionMNIST("./Dataset/FashionMNIST/test", train=False, download=True, transform=img_transform)
        print("Test Train Data:", test_set.data.shape)

    elif config['dataset_name'] in ['mvtec']:
        data_path = 'Dataset/MVTec/' + normal_class + '/train'
        mvtec_img_size = config['mvtec_img_size']

        orig_transform = transforms.Compose([
            transforms.Resize([mvtec_img_size, mvtec_img_size]),
            transforms.ToTensor()
        ])

        dataset = ImageFolder(root=data_path, transform=orig_transform)

        test_data_path = 'Dataset/MVTec/' + normal_class + '/test'
        test_set = ImageFolder(root=test_data_path, transform=orig_transform)

    elif config['dataset_name'] in ['retina']:
        data_path = 'Dataset/OCT2017/train'

        orig_transform = transforms.Compose([
            transforms.Resize([128, 128]),
            transforms.ToTensor()
        ])

        dataset = ImageFolder(root=data_path, transform=orig_transform)

        test_data_path = 'Dataset/OCT2017/test'
        test_set = ImageFolder(root=test_data_path, transform=orig_transform)

    else:
        raise Exception(
            "You enter {} as dataset, which is not a valid dataset for this repository!".format(config['dataset_name']))

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_dataloader, test_dataloader


def load_localization_data(config):
    normal_class = config['normal_class']
    mvtec_img_size = config['mvtec_img_size']

    orig_transform = transforms.Compose([
        transforms.Resize([mvtec_img_size, mvtec_img_size]),
        transforms.ToTensor()
    ])

    test_data_path = 'Dataset/MVTec/' + normal_class + '/test'
    test_set = ImageFolder(root=test_data_path, transform=orig_transform)
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=512,
        shuffle=False,
    )

    ground_data_path = 'Dataset/MVTec/' + normal_class + '/ground_truth'
    ground_dataset = ImageFolder(root=ground_data_path, transform=orig_transform)
    ground_dataloader = torch.utils.data.DataLoader(
        ground_dataset,
        batch_size=512,
        num_workers=0,
        shuffle=False
    )

    x_ground = next(iter(ground_dataloader))[0].numpy()
    ground_temp = x_ground

    std_groud_temp = np.transpose(ground_temp, (0, 2, 3, 1))
    x_ground = std_groud_temp

    return test_dataloader, x_ground
