import os
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import argparse
from tqdm import tqdm
from utils.distill import Attention, DistillKL
from utils.metric import EvaluatorSeg
from utils.dataset import make_datapath_seg_list, VOCSegDataset
from utils.efficientdet import EfficientDet

def str2bool(s):
    if s not in {'F', 'T'}:
        raise ValueError('Not a valid boolean string')
    return s == 'T'

def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--random_seed', default=1, type=int)
parser.add_argument('--data_dir', default='', type=str)
parser.add_argument('--num_epochs', default=60, type=int)
parser.add_argument('--lr', default=0.01, type=float)

parser.add_argument('--backbone', default='0', type=str)
parser.add_argument('--batch_size', default=2, type=int)

parser.add_argument('--alpha', default=0, type=float)
parser.add_argument('--beta', default=0, type=float)

parser.add_argument('--step', default=[40], type=int, nargs='+')
parser.add_argument('--warmup', default=40, type=int)
parser.add_argument('--anneal', default='T', type=str2bool)
parser.add_argument('--temp', default=1, type=float)

test_break = False
server = True
if not server:
    vocpath = "/data/Image/Detection/voc/VOCdevkit"
else:
    vocpath = "../VOCdevkit"

args = parser.parse_args()
args.data_dir = vocpath
args.log_path = 'b{}_a{}_b{}_w{}_a{}_t{}'.format(args.backbone, args.alpha, args.beta, args.warmup, args.anneal, args.temp)
random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
cudnn.deterministic = True

backbone = "efficientnet-b%s" % args.backbone
accum_step = 32.0 / args.batch_size

train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_seg_list(os.path.join(args.data_dir, 'VOC2007'))
train_img_list2, train_anno_list2, _, _ = make_datapath_seg_list(os.path.join(args.data_dir, 'VOC2012'))

train_img_list.extend(train_img_list2)
train_anno_list.extend(train_anno_list2)

print("trainlist: ", len(train_img_list))
print("vallist: ", len(val_img_list))

# make Dataset
train_dataset = VOCSegDataset(train_img_list, train_anno_list, phase="train")
val_dataset = VOCSegDataset(val_img_list, val_anno_list, phase="val")
train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                          num_workers=4, pin_memory=True)
valid_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
dataloaders_dict = {"train": train_loader, "val": valid_loader}

batch_iterator = iter(dataloaders_dict["val"])
num_class = 21

sample = next(batch_iterator)
print(sample['image'].size())
print(sample['label'].size())

net = EfficientDet(backbone=backbone)
criterion_att = Attention()
criterion_kd = DistillKL(args)
criterion = nn.CrossEntropyLoss(ignore_index=255, size_average=True)
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-3, momentum=0.9, weight_decay=5e-4)


def get_current_lr(epoch, lr, step=[40, 60]):
    for i, lr_decay_epoch in enumerate(step):
        if epoch >= lr_decay_epoch:
            lr *= 0.1
    return lr


def adjust_learning_rate(optimizer, epoch, lr, step):
    lr = get_current_lr(epoch, lr, step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_model_seg(net, dataloaders_dict, criterion, optimizer, num_epochs, args):
    net.cuda()
    logs = []

    for epoch in range(num_epochs):
        start = time.time()
        total_loss = 0.0
        net.train()
        tbar = tqdm(dataloaders_dict['train'])

        # train
        if epoch < args.warmup:
            beta = 0
            alpha = 0
        else:
            if args.anneal:
                beta = args.beta * sigmoid_rampup(epoch+1 - args.warmup, num_epochs - args.warmup)
                alpha = args.alpha * sigmoid_rampup(epoch+1 - args.warmup, num_epochs - args.warmup)
            else:
                beta = args.beta
                alpha = args.alpha


        adjust_learning_rate(optimizer, epoch, args.lr, args.step)
        for i, sample in enumerate(tbar):
            image, target = sample['image'].cuda(), sample['label'].cuda()
            if args.alpha > 0:
                out1, out2, sources1, sources2 = net(image)
                loss_kd = beta * criterion_att(sources1, sources2)
                loss_kd += alpha * (criterion_kd(out1, out2))

                loss = (criterion(out1, target.long()) + criterion(out2, target.long()) + loss_kd) / accum_step
            else:
                out1, _, _, _ = net(image)
                loss = criterion(out1, target.long()) / accum_step
            loss.backward()
            if (i+1) % accum_step == 0:
                nn.utils.clip_grad_value_(net.parameters(), clip_value=2.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            if i % 10:
                tbar.set_description('Train loss: %.3f' % (total_loss / (i + 1)))
            if test_break:
                break
        total_loss = total_loss / (i + 1)
        time_spend = time.time() - start

        net.eval()
        evaluator = EvaluatorSeg(num_class=21)
        tbar = tqdm(dataloaders_dict['val'])

        for i, sample in enumerate(tbar):
            image, target = sample['image'].cuda(), sample['label'].cuda()
            with torch.no_grad():
                output, _, _, _ = net(image)
                pred = output.data.cpu().numpy()
                target = target.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                evaluator.add_batch(target, pred)
            if test_break:
                break
        mIoU = evaluator.Mean_Intersection_over_Union()
        log_epoch = {'epoch': epoch + 1, 'train_loss': total_loss, 'mIOU': mIoU, 'time': time_spend}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("logs/{}.txt".format(args.log_path))


train_model_seg(net, dataloaders_dict, criterion, optimizer, num_epochs=args.num_epochs, args=args)