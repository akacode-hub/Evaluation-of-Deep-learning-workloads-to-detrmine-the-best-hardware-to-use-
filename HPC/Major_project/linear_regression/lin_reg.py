import sys, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

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

    print("num pca components: ", pca.n_components_)
    X_train_proc = pca.transform(train_data)
    X_test_proc = pca.transform(X_test_std)

    return 

if __name__ == "__main__":

    fpath = '../dataset/YearPredictionMSD.txt'
    num_train = 463715

    features, labels = get_data(fpath)    
    data = train_test_split(features, labels, num_train)
    train_data, train_labels = data[:2]
    test_data, test_labels = data[2:]

    scale_train_data, scale_test_data = scale_data(train_data, test_data)

