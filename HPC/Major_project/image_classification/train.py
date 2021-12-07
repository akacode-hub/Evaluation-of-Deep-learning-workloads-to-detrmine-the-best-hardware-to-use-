import os, sys
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import time
import torch.nn.functional as F
from torchvision.datasets import CIFAR100
import torchvision.transforms as tt
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split,ConcatDataset
from model import MResnet

def get_data():

    stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
    train_transform = tt.Compose([
        tt.RandomHorizontalFlip(),
        tt.RandomCrop(32,padding=4),
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
        label = train_item[1]
        if label not in train_classes_items:
            train_classes_items[label] = 1
        else:
            train_classes_items[label] += 1

    test_classes_items = dict()
    for test_item in test_data:
        label = test_item[1]
        if label not in test_classes_items:
            test_classes_items[label] = 1
        else:
            test_classes_items[label] += 1

    return train_classes_items, test_classes_items

def show_batch(data_loader):

    for batch in data_loader:
        images,labels = batch
        fig, ax = plt.subplots(figsize=(7.5,7.5))
        ax.set_yticks([])
        ax.set_xticks([])
        ax.imshow(make_grid(images[:20],nrow=5).permute(1,2,0))
        break

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def to_device(data,device):
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)

class ToDeviceLoader:
    def __init__(self,data,device):
        self.data = data
        self.device = device
        
    def __iter__(self):
        for batch in self.data:
            yield to_device(batch,self.device)
            
    def __len__(self):
        return len(self.data)

@torch.no_grad()
def evaluate(model,test_data_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in test_data_loader]
    return model.validation_epoch_end(outputs)

def train():

    train_data, test_data = get_data()
    train_cls_dist, test_cls_dist = class_dist(train_data, test_data)
    print('train_cls_dist: ',train_cls_dist)
    print('test_cls_dist: ',test_cls_dist)

    train_data_loader = DataLoader(train_data, batch_size, num_workers=num_workers,pin_memory=True,shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size, num_workers=num_workers,pin_memory=True)

    #show_batch(train_data_loader)
    device = get_device()

    train_data_loader = ToDeviceLoader(train_data_loader, device)
    test_data_loader = ToDeviceLoader(test_data_loader, device)

    model = MResnet(3, 100)
    model = to_device(model, device)

    optimizer = torch.optim.Adam(model.parameters(), lr,weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=lr_drop)

    for epoch in range(0, num_epochs):

        print('Epoch:', epoch, 'lr: ', scheduler.get_last_lr())
        losses = []
        start = time.time()

        for batch in train_data_loader:

            loss = model.training_step(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.data.item())
            
        scheduler.step()
        train_mins = (time.time() - start) / 60 
        losses_np = np.mean(np.asarray(losses))

        print('Epoch [{}/{}], time: {:.4f} mins, lr: {}, loss: {:.4f}'.format(epoch + 1, num_epochs, train_mins, scheduler.get_last_lr()[0], losses_np), flush=True)

        if epoch % 5 ==0:
            result = evaluate(model, test_data_loader)
            val_loss = result["val_loss"]
            val_acc = result["val_acc"]
            print('Epoch [{}/{}], val_loss: {:.4f}, val_acc: {:.4f}'.format(epoch + 1, num_epochs, val_loss, val_acc), flush=True)
            torch.save(model.state_dict(), model_save_dir + str(epoch) + '.pth')

if __name__ == "__main__":

    batch_size = 512
    num_workers = 24
    num_epochs = 100
    lr = 1e-3
    grad_clip = 0.1
    weight_decay = 1e-4
    lr_steps = [20, 40, 60, 80]
    lr_drop = 0.2
    model_save_dir = 'models/exp1/'

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    train()