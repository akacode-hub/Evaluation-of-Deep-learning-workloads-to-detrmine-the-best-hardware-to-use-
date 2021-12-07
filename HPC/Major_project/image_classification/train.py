import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR100
import torchvision.transforms as tt
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split,ConcatDataset

def get_data():

    stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
    train_transform = tt.Compose([
        tt.RandomHorizontalFlip(),
        tt.RandomCrop(32,padding=4,padding_mode="reflect"),
        tt.ToTensor(),
        tt.Normalize(*stats)
    ])

    test_transform = tt.Compose([
        tt.ToTensor(),
        tt.Normalize(*stats)
    ])

    train_data = CIFAR100(download=True,root="./data",transform=train_transform)
    test_data = CIFAR100(root="./data",train=False,transform=test_transform)

    return train_data, test_data

def class_dist(train_data, test_data):

    train_classes_items = dict()

    for train_item in train_data:
        label = train_data.classes[train_item[1]]
        if label not in train_classes_items:
            train_classes_items[label] = 1
        else:
            train_classes_items[label] += 1

    test_classes_items = dict()
    for test_item in test_data:
        label = test_data.classes[test_item[1]]
        if label not in test_classes_items:
            test_classes_items[label] = 1
        else:
            test_classes_items[label] += 1

    return train_classes_items, test_classes_items

def show_batch(dl):
    for batch in dl:
        images,labels = batch
        fig, ax = plt.subplots(figsize=(7.5,7.5))
        ax.set_yticks([])
        ax.set_xticks([])
        ax.imshow(make_grid(images[:20],nrow=5).permute(1,2,0))
        break

def train():

    train_data, test_data = get_data()
    train_cls_dist, test_cls_dist = class_dist(train_data, test_data)
    print('train_cls_dist: ',train_cls_dist)
    print('test_cls_dist: ',test_cls_dist)

if __name__ == "__main__":

    train()