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

class YearPredictionDataset(Dataset):
    
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

def train_test_split(features, labels, num_train):

    train_feats = features[:num_train]
    train_labels = labels[:num_train]
    
    test_feats = features[num_train:]
    test_labels = labels[num_train:]

    return [train_feats, train_labels, test_feats, test_labels]

def scale_data(train_data, test_data):

    scaler = StandardScaler()

    scaler.fit(train_data)
    
    scale_train_data = scaler.transform(train_data)
    scale_test_data = scaler.transform(test_data)

    return scale_train_data, scale_test_data

def scale_labels(train_labels, test_labels):

    min_label, max_label = np.min(train_labels), np.max(train_labels)
    width = max_label - min_label
    
    train_labels = train_labels - min_label
    test_labels = test_labels - min_label

    return train_labels, test_labels

def dimension_reduction(train_data, test_data, variance_frac=0.9):

    pca = PCA(variance_frac)
    pca.fit(train_data)

    X_train_proc = pca.transform(train_data)
    X_test_proc = pca.transform(test_data)

    return X_train_proc, X_test_proc, pca.n_components_


class MLP(nn.Module):

    def __init__(self, num_channels, out_size):

        super(MLP, self).__init__()

        # Hidden layers
        self.layers = []
        self.layers.append(nn.Linear(num_channels, 2*num_channels))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(2*num_channels, 2*num_channels))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(2*num_channels, num_channels))
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
        return self.network(x)

def calc_mae(output, label):

    return torch.mean(torch.abs(output - label))

def train_gpu(num_dim, gpu_id=0):

    features, labels = get_data(fpath)    
    data = train_test_split(features, labels, num_train)
    train_data, train_labels = data[:2]
    test_data, test_labels = data[2:]
    scale_train_data, scale_test_data = scale_data(train_data, test_data)

    train_labels, test_labels = scale_labels(train_labels, test_labels)

    if use_pca:
        scale_train_data, scale_test_data, num_dim = dimension_reduction(scale_train_data, scale_test_data)
        print('pca dim: ', num_dim)

    torch.manual_seed(0)
    model = MLP(num_dim, pred_dim)
    print('model: ',model)
    torch.cuda.set_device(gpu_id)
    model.cuda(gpu_id)
    
    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss().cuda(gpu_id)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=lr_drop)

    # Data loading code
    train_dataset = YearPredictionDataset(features=scale_train_data, labels=train_labels)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers,
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
            metric = calc_mae(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        train_mins = (time.time() - start) / 60 
        print('Epoch [{}/{}], time: {:.4f} mins, lr: {}, mse: {:.4f},  mae: {:.4f}'.format(epoch + 1, num_epochs, train_mins, scheduler.get_last_lr()[0], loss.item(), metric), flush=True)

        if epoch % 2 == 0:
            torch.save(model.state_dict(), model_save_dir + str(epoch) + '.pth')

def train_cpu(num_dim):

    features, labels = get_data(fpath)    
    data = train_test_split(features, labels, num_train)
    train_data, train_labels = data[:2]
    test_data, test_labels = data[2:]
    scale_train_data, scale_test_data = scale_data(train_data, test_data)

    train_labels, test_labels = scale_labels(train_labels, test_labels)

    if use_pca:
        scale_train_data, scale_test_data, num_dim = dimension_reduction(scale_train_data, scale_test_data)
        print('pca dim: ', num_dim)

    torch.manual_seed(0)
    model = MLP(num_dim, pred_dim)
    print('model: ',model)
    
    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=lr_drop)

    # Data loading code
    train_dataset = YearPredictionDataset(features=scale_train_data, labels=train_labels)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               pin_memory=True)


    total_step = len(train_loader)

    for epoch in range(num_epochs):

        start = time.time()
        for i, sample in enumerate(train_loader):
            
            features = sample['features'].float()
            labels = sample['label'].float()

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            metric = calc_mae(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        train_mins = (time.time() - start) / 60 
        print('Epoch [{}/{}], time: {:.4f} mins, lr: {}, mse: {:.4f},  mae: {:.4f}'.format(epoch + 1, num_epochs, train_mins, scheduler.get_last_lr()[0], loss.item(), metric), flush=True)

        if epoch % 2 == 0:
            torch.save(model.state_dict(), model_save_dir + str(epoch) + '.pth')

def validate(network, num_dim):

    features, labels = get_data(fpath)    
    data = train_test_split(features, labels, num_train)
    train_data, train_labels = data[:2]
    test_data, test_labels = data[2:]
    scale_train_data, scale_test_data = scale_data(train_data, test_data)

    if use_pca:
        scale_train_data, scale_test_data, num_dim = dimension_reduction(scale_train_data, scale_test_data)

    train_labels, test_labels = scale_labels(train_labels, test_labels)

    mean_mae = 0.0    
    tot_pred_time = 0.0
    for idx in range(scale_test_data.shape[0]):
        
        test_inp = scale_test_data[idx]
        test_label = test_labels[idx]
        test_inp = torch.tensor(test_inp).float()

        start = time.time()
        pred = network(test_inp)
        end = time.time()
        tot_pred_time += end - start

        pred = pred.detach().numpy()
        mae_err = np.abs(pred - test_label)

        test_label = int(test_label + 1922)
        pred = int(pred + 1922)

        #print('idx: ',idx, ' label: ',test_label, ' pred: ', pred, ' mse_err: ',mse_err)
        mean_mae += mae_err
    
    return np.round(mean_mae/scale_test_data.shape[0], 3)[0], np.round(tot_pred_time, 3)

def test(model_path, num_dim):

    if not usegpu:
        print('Using CPU for testing')
    else:
        print('Using GPU for testing')

    print('model_path: ',model_path)
    network = MLP(num_dim, pred_dim)

    if usegpu:
        network.load_state_dict(torch.load(model_path))
    else:
        network.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    network.eval()

    start = time.time()
    avg_mae, pred_time = validate(network, num_dim)
    print('avg_mae: ',avg_mae)
    end = time.time()
    test_secs = (time.time() - start)
    print('Test Mean MAE for {} samples: {:.4f}'.format(num_test, avg_mae), flush=True)
    print('Total Prediction Time elapsed to test {} samples: {:.4f} secs'.format(num_test, pred_time), flush=True)
    print('Total Time elapsed to test {} samples: {:.4f} secs'.format(num_test, test_secs), flush=True)

def print_params():

    print('num_train: ',num_train)
    print('batch_size: ',batch_size)
    print('num_epochs: ',num_epochs)
    print('num_dim: ',num_dim)
    print('pred_dim: ',pred_dim)
    print('use_pca: ',use_pca)
    print('num_workers: ',num_workers)
    print('lr: ',lr)
    print('lr_steps: ',lr_steps)
    print('lr_drop: ',lr_drop, flush=True)

if __name__ == "__main__":

    fpath = '../dataset/YearPredictionMSD.txt'
    model_save_dir = 'models/exp1/'

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    num_train = 463715
    num_test = 51630
    batch_size = 512
    num_epochs = 102
    num_dim = 90
    pred_dim = 1
    use_pca = 0
    lr = 5e-3
    lr_steps = [50]
    lr_drop = 0.1
    num_workers = 0

    if not torch.cuda.is_available():
        usegpu = 0
    else:
        usegpu = 1

    print_params()

    #train    

    if usegpu:
        print('Using GPU for training')
        train_gpu(num_dim)
    else:
        print('Using CPU for training')
        train_cpu(num_dim)

    #validate
    model_path = os.path.join(model_save_dir, '100.pth')
    test(model_path, num_dim)