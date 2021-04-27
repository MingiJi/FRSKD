import os
import torch
import random
import numpy as np
from collections import defaultdict
from torchvision import transforms, datasets
from torchvision.transforms.functional import rotate
from torch.utils.data import Sampler, BatchSampler, RandomSampler


def create_loader(batch_size, data_dir, data):
    loader = {'CIFAR100': cifar_loader, 'TINY': tiny_loader, 'otherwise': imageset_loader}
    load_data = data if data in ['CIFAR100', 'CIFAR10', 'TINY'] else 'otherwise'
    return loader[load_data](batch_size, data_dir, data)


def default_sampler(dataset, batch_size):
    return BatchSampler(RandomSampler(dataset), batch_size, False)


def cifar_loader(batch_size, data_dir, data):
    num_label = 100
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,])
    transform_test = transforms.Compose([transforms.ToTensor(), normalize, ])

    trainset = datasets.CIFAR100(root=os.path.join(data_dir, data), train=True,
                                 download=False, transform=transform_train)
    testset = datasets.CIFAR100(root=os.path.join(data_dir, data), train=False,
                                download=False, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader, num_label


def tiny_loader(batch_size, data_dir, data):
    num_label = 200
    normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
    transform_train = transforms.Compose(
        [transforms.RandomResizedCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         normalize, ])
    transform_test = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize, ])
    trainset = datasets.ImageFolder(root=os.path.join(data_dir, data, 'train'), transform=transform_train)
    testset = datasets.ImageFolder(root=os.path.join(data_dir, data, 'valid'), transform=transform_test)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader, num_label


def imageset_loader(batch_size, data_dir, data):
    if data.lower() == 'cub_200_2011':
        num_label = 200
    elif data.lower() == 'dogs':
        num_label = 120
    elif data.lower() == 'mit67':
        num_label = 67
    elif data.lower() == 'stanford40':
        num_label = 40
    elif data.lower() == 'imagenet':
        num_label = 1000
    else:
        raise NotImplementedError('Dataset {} is not prepared.'.format(data))
    kwargs = {'num_workers': 4, 'pin_memory': True}
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform_train = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(), normalize, ])
    transform_test = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                         transforms.ToTensor(), normalize, ])
    trainset = datasets.ImageFolder(root=os.path.join(data_dir, data, 'train'), transform=transform_train)
    testset = datasets.ImageFolder(root=os.path.join(data_dir, data, 'valid'), transform=transform_test)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader, num_label