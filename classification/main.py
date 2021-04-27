import os
import random
import argparse
import torch.nn as nn
from time import time
from classification.dataset import create_loader
from classification import models, distill_loss
from classification.utils import *


def str2bool(s):
    if s not in {'F', 'T'}:
        raise ValueError('Not a valid boolean string')
    return s == 'T'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--data', default='CIFAR100', type=str)
    parser.add_argument('--random_seed', default=10, type=int)

    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--scheduler', default='step', type=str, help='step|cos')
    parser.add_argument('--schedule', default=[100, 150], type=int, nargs='+')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--lr_decay', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    parser.add_argument('--model', default='cifarresnet18', type=str)

    parser.add_argument('--num_channels', default=256, type=int)
    parser.add_argument('--num_features', default=-1, type=int)

    parser.add_argument('--repeat', default=1, type=int)
    parser.add_argument('--depth', default=2, type=int)
    parser.add_argument('--temperature', default=4, type=float)

    # distill
    parser.add_argument('--bifpn', default='BiFPNc', type=str, help='BiFPN|BiFPNc')
    parser.add_argument('--width', default=2, type=int)
    parser.add_argument('--distill', default='att', type=str)
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--beta', default=0.0, type=float)
    parser.add_argument('--aux', default='none', type=str)
    parser.add_argument('--aux_lamb', default=0.0, type=float)

    # augmentation
    parser.add_argument('--aug', default='none', type=str)
    parser.add_argument('--aug_a', default=0.0, type=float)

    args = parser.parse_args()
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args_path = '{}_{}_{}_a{}_b{}_{}{}_{}{}'.format(args.data, args.model, args.distill, args.alpha, args.beta, args.aux, args.aux_lamb, args.aug, args.aug_a)
    path_log = os.path.join('logs', args_path)
    if not os.path.exists(path_log):
        os.makedirs(path_log)
    logger = create_logging(os.path.join(path_log, '%s.txt' % args.random_seed))
    train_loader, test_loader, args.num_classes = create_loader(args.batch_size, args.data_dir, args.data)

    for param in sorted(vars(args).keys()):
        logger.info('--{0} {1}'.format(param, vars(args)[param]))

    args.depth = [args.depth] * 3
    model = models.__dict__[args.model](num_classes=args.num_classes)
    if args.num_features == -1:
        args.num_features = len(model.network_channels)
    args.network_channels = model.network_channels[-args.num_features:]
    bifpn = models.__dict__[args.bifpn](args.network_channels, args.num_classes, args)

    if args.aux == 'sla':
        criterion_ce = distill_loss.__dict__[args.aux](args)
        criterion_ce.train()
    else:
        criterion_ce = nn.CrossEntropyLoss()

    criterion_kd = distill_loss.__dict__[args.distill](args, bifpn)
    criterion_kd.train()
    train_list = nn.ModuleList()
    train_list.append(model)
    train_list.append(criterion_ce)
    train_list.append(criterion_kd)
    train_list.append(bifpn)
    bifpn.cuda()
    train_list.cuda()

    criterion = [criterion_ce, criterion_kd]
    optimizer = optim.SGD(train_list.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = lr_scheduler(optimizer, args.scheduler, args.schedule, args.lr_decay, args.epoch)

    for epoch in range(1, args.epoch+1):
        s = time()
        loss, train_acc1 = train(model, bifpn, optimizer, criterion, train_loader, args)
        scheduler.step()
        test_acc1 = test(model, test_loader)
        logger.info('Epoch: {0:>2d}|Train Loss: {1:2.4f}| Train Acc: {2:.4f}| Test Acc: {3:.4f}| Time: {4:4.2f}(s)'
                    .format(epoch, loss, train_acc1, test_acc1, time() - s))


if __name__ == '__main__':
    main()