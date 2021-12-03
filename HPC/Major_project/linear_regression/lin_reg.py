import sys, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

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

    scaler = MinMaxScaler()

    scaler.fit(train_data)
    
    scale_train_data = scaler.transform(train_data)
    scale_test_data = scaler.transform(test_data)

    return scale_train_data, scale_test_data

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
        self.layers.append(nn.Linear(num_channels, num_channels))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Linear(num_channels, out_size))

        self.network = nn.Sequential(*self.layers)
        self.init_weights()

    def init_weights(self, gate_bias_init: float = 0.0) -> None:

        for i in [0, 2]:

            fc = self.layers[i]
            torch.nn.init.xavier_uniform_(fc.weight)
            fc.bias.data.fill_(gate_bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Feedforward
        return self.network(x)

def train(num_dim, gpu_id=0):

    features, labels = get_data(fpath)    
    data = train_test_split(features, labels, num_train)
    train_data, train_labels = data[:2]
    test_data, test_labels = data[2:]
    scale_train_data, scale_test_data = scale_data(train_data, test_data)

    if use_pca:
        scale_train_data, scale_test_data, num_dim = dimension_reduction(scale_train_data, scale_test_data)

    torch.manual_seed(0)
    model = MLP(num_dim, pred_dim)
    torch.cuda.set_device(gpu_id)
    model.cuda(gpu_id)
    
    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss().cuda(gpu_id)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    # Data loading code
    train_dataset = YearPredictionDataset(features=scale_train_data, labels=train_labels)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True)


    total_step = len(train_loader)

    for epoch in range(num_epochs):

        for i, sample in enumerate(train_loader):
            
            features = sample['features'].float()
            labels = sample['label'].float()
            features = features.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

        if epoch % 5 == 0:
            torch.save(model.state_dict(), model_save_dir + str(epoch) + '.pth')

if __name__ == "__main__":

    fpath = '../dataset/YearPredictionMSD.txt'
    model_save_dir = 'models/exp1/'

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    num_train = 463715
    batch_size = 100
    num_epochs = 100
    num_dim = 90
    pred_dim = 1
    use_pca = 0
    
    train(num_dim)