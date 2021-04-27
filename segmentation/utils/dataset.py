import os
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import utils.custom_transform_seg as tr
from torchvision import transforms


def make_datapath_seg_list(rootpath):
    img_path_template = os.path.join(rootpath, "JPEGImages", "%s.jpg")
    anno_path_template = os.path.join(rootpath, "SegmentationClass", "%s.png")

    # get id
    train_id_names = os.path.join(rootpath, "ImageSets", "Segmentation", "trainval.txt")
    val_id_names = os.path.join(rootpath, "ImageSets", "Segmentation", "test.txt")

    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names):
        file_id = line.strip()
        img_path = (img_path_template % file_id)
        anno_path = (anno_path_template % file_id)
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    val_img_list = list()
    val_anno_list = list()
    if os.path.isfile(val_id_names):
        for line in open(val_id_names):
            file_id = line.strip()
            img_path = (img_path_template % file_id)
            anno_path = (anno_path_template % file_id)
            val_img_list.append(img_path)
            val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list


class VOCSegDataset(data.Dataset):
    def __init__(self, img_list, anno_list, phase):
        self.img_list = img_list
        self.anno_list = anno_list
        self.split = phase

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}
        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.img_list[index]).convert('RGB')
        _target = Image.open(self.anno_list[index])
        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=512, crop_size=512),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixedResize(size=512),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)