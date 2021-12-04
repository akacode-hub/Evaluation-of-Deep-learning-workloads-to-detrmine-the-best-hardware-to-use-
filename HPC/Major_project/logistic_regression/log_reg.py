import sys, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import time

class HIGGSDataset(Dataset):
    
    def __init__(self, features, labels):
        
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        
        sample = {'features': self.features[idx], 'label': self.labels[idx]}

        return sample

def get_data(filename, num_rows=None):
    
    year = pd.read_csv(filename, header=None, nrows=num_rows)
    X = year.iloc[:, 1:].values
    y = year.iloc[:, 0].values

    return X, y

def scale_data(train_data, test_data):

    scaler = StandardScaler()

    scaler.fit(train_data)
    
    scale_train_data = scaler.transform(train_data)
    scale_test_data = scaler.transform(test_data)

    return scale_train_data, scale_test_data

def train_test_split(features, labels, num_train):

    train_feats = features[:num_train]
    train_labels = labels[:num_train]
    
    test_feats = features[num_train:]
    test_labels = labels[num_train:]

    return [train_feats, train_labels, test_feats, test_labels]

def print_params():

    print('num_train: ',num_train)
    print('batch_size: ',batch_size)
    print('num_epochs: ',num_epochs)
    print('num_dim: ',num_dim)
    print('pred_dim: ',pred_dim)

    print('lr: ',lr)
    print('lr_steps: ',lr_steps)
    print('lr_drop: ',lr_drop, flush=True)

class MLP(nn.Module):

    def __init__(self, num_channels, out_size):

        super(MLP, self).__init__()
        self.sigmoid = nn.Sigmoid()
        
        # Hidden layers
        self.layers = []
        self.layers.append(nn.Linear(num_channels, num_channels))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(num_channels, out_size))

        self.network = nn.Sequential(*self.layers)
        self.init_weights()

    def init_weights(self, gate_bias_init: float = 0.0) -> None:

        for i in range(0, len(self.layers), 2):

            fc = self.layers[i]
            torch.nn.init.xavier_uniform_(fc.weight)
            fc.bias.data.fill_(gate_bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Feedforward
        out = self.network(x)

        return self.sigmoid(out)

def train(num_dim, gpu_id=0):

    features, labels = get_data(fpath)    
    data = train_test_split(features, labels, num_train)
    train_data, train_labels = data[:2]
    test_data, test_labels = data[2:]
    scale_train_data, scale_test_data = scale_data(train_data, test_data)

    torch.manual_seed(0)
    model = MLP(num_dim, pred_dim)
    print('model: ',model)
    torch.cuda.set_device(gpu_id)
    model.cuda(gpu_id)

    # define loss function (criterion) and optimizer
    criterion = nn.BCELoss().cuda(gpu_id)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=lr_drop)

    # Data loading code
    train_dataset = HIGGSDataset(features=scale_train_data, labels=train_labels)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True)

    total_step = len(train_loader)

    for epoch in range(num_epochs):

        start = time.time()
        for i, sample in enumerate(train_loader):
            
            features = sample['features'].float()
            labels = sample['label'].float()
            
            features = features.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            metric = 0

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        train_mins = (time.time() - start) / 60 
        print('Epoch [{}/{}], time: {:.4f} mins, lr: {}, mse: {:.4f},  mae: {:.4f}'.format(epoch + 1, num_epochs, train_mins, scheduler.get_last_lr()[0], loss.item(), metric), flush=True)

        if epoch % 2 == 0:
            torch.save(model.state_dict(), model_save_dir + str(epoch) + '.pth')
            
if __name__ == "__main__":

    fpath = '../dataset/HIGGS.csv'
    model_save_dir = 'models/exp1/'

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    num_train = 10500000
    batch_size = 512
    num_epochs = 102
    pred_dim = 1
    num_dim = 28

    lr = 5e-3
    lr_steps = [50]
    lr_drop = 0.1

    print_params()

    #train
    train(num_dim)

    #validate
    # model_path = os.path.join(model_save_dir, '100.pth')
    # test(model_path, num_dim)
