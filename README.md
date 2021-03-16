# FRSKD
Official implementation for [Refine Myself by Teaching Myself : Feature Refinement via Self-Knowledge Distillation](https://arxiv.org/abs/2103.08273) (CVPR-2021)

## Requirements
- Python3
- Pytorch (>1.4.0)
- torchvision
- numpy 
- Pillow
- tqdm

## Training
In this code, you can reproduce the experiment results of classification task in submitted paper.
The datasets are all open-sourced, so it is easy to download.
Example training settings are for ResNet18 on CIFAR-100.
Detailed hyperparameter settings are enumerated in the paper.
- Training with FRSKD
~~~
python main.py --data_dir PATH_TO_DATASET \
--data CIFAR100 --batch_size 128 --alpha 2 --beta 100 \
--aux none --aux_lamb 0 --aug none --aug_a 0
~~~
- Training with FRSKD + SLA
~~~
python main.py --data_dir PATH_TO_DATASET \
--data CIFAR100 --batch_size 128 --alpha 2 --beta 100 \
--aux sla --aux_lamb 1 --aug none --aug_a 0
~~~
- Training with FRSKD + Mixup
~~~
python main.py --data_dir PATH_TO_DATASET \
--data CIFAR100 --batch_size 128 --alpha 2 --beta 100 \
--aux none --aux_lamb 0 --aug mixup --aug_a 0.2
~~~
- Training with FRSKD + CutMix
~~~
python main.py --data_dir PATH_TO_DATASET \
--data CIFAR100 --batch_size 128 --alpha 2 --beta 100 \
--aux none --aux_lamb 0 --aug cutmix --aug_a 1.0
~~~