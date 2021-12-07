import os, sys
import numpy as np
import random
import time
import torch
np.set_printoptions(suppress=True)
from torch import nn, load, save
from torch.optim import lr_scheduler, Adam
import torch.utils.data as dataset
from torch.utils.data import DataLoader
from data_loader import train_data_loader
from mobilenet import V2_m
import cv2

def get_dataloader():

    train_data = train_data_loader.DriverDataset(train_data_path, train=True)
    train_dataloader = DataLoader(dataset=train_data,shuffle=True, batch_size=batch_size, num_workers=num_workers)

    val_data = train_data_loader.DriverDataset(train_data_path, train=False)
    val_dataloader = DataLoader(dataset=val_data, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    return train_dataloader, val_dataloader

def get_model():

    model = V2_m.mobilenet_v2()
    model.cuda()

    return model

def train():

    train_dataloader, val_dataloader = get_dataloader()

    criterion = nn.CrossEntropyLoss()

    model = get_model()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.001)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=lr_drop)

    for epoch in range(0, num_epochs):

        print('Epoch:', epoch, 'lr: ', scheduler.get_last_lr())
        losses = []
        start = time.time()

        for (data, label) in train_dataloader:

            data = data.cuda()
            label = label.cuda()

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.data.item())
        
    
        scheduler.step()
        losses_np = np.mean(np.asarray(losses))

        train_mins = (time.time() - start) / 60 
        print('Epoch [{}/{}], time: {:.4f} mins, lr: {}, loss: {:.4f}'.format(epoch + 1, num_epochs, train_mins, scheduler.get_last_lr()[0], losses_np), flush=True)

        torch.save(model.state_dict(), model_save_dir + str(epoch) + '.pth')

def validate(vis=0):

    train_dataloader, val_dataloader = get_dataloader()
    
    print('model_path: ',model_path)
    model = get_model()

    model.load_state_dict(torch.load(model_path))
    model.eval()
    accs = []

    for i, (val_input, val_label) in enumerate(val_dataloader):

        val_input = val_input.cuda()
        val_label = val_label.cuda()

        output = model(val_input)
        output = torch.max(output, 1)[1]

        acc = torch.mean(torch.eq(output, val_label).float())

        output = output.cpu().data.numpy()[0]
        val_label = val_label.cpu().data.numpy()[0]
        acc = acc.cpu().data.numpy()

        val_input = val_input.cpu().data.numpy()[0]
        val_input = np.transpose(val_input, (1, 2, 0))

        accs.append(acc)

        if vis:
            print('output: ', output)
            print('val_label: ', val_label)
            print('acc: ', acc)
            cv2.imshow('val_input ', val_input)
            cv2.waitKey(-1)

    mean_acc = np.mean(np.array(accs))
    mean_acc = np.round(mean_acc, 3)
    print('mean_acc: ', mean_acc)

if __name__ == "__main__":

    train_data_path = '/home/balaji/Documents/code/RSL/NEU-Fall2021/MLPR/Project/state-farm-distracted-driver-detection/imgs/train/'
    batch_size = 1
    num_epochs = 100

    lr = 1e-4
    lr_steps = [30, 60, 90, 120]
    lr_drop = 0.1
    num_workers = 8
    model_save_dir = 'models/exp1/'

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    train = 0

    if train:
        train()

    else:
        model_path = os.path.join(model_save_dir, '90.pth')
        validate()