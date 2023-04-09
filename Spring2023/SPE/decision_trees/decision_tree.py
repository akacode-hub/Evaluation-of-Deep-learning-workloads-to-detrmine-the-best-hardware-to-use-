import csv
import numpy as np
import os.path
import pandas as pd
import time
import xgboost as xgb
import sys
np.set_printoptions(suppress=True)

def get_data(filename):
    
    higgs_train = pd.read_csv(filename, dtype=np.float32, 
                                     nrows=train_rows, header=None)

    higgs_test = pd.read_csv(filename, dtype=np.float32, 
                                    skiprows=train_rows, nrows=test_rows, 
                                    header=None)
    
    train_label = higgs_train.iloc[:, 0].values
    test_label = higgs_test.iloc[:, 0].values

    higgs_train = xgb.DMatrix(higgs_train.loc[:, 1:29], higgs_train[0])
    higgs_test = xgb.DMatrix(higgs_test.loc[:, 1:29], higgs_test[0])

    return higgs_train, higgs_test, train_label, test_label

def calc_metric(output, label, thresh=0.5):

    output = (output>thresh).astype('int')

    return 1 - np.mean(np.abs(output - label))

def train_GPU():

    param = {}
    param['objective'] = 'binary:logitraw'
    param['eval_metric'] = 'error'
    param['tree_method'] = 'gpu_hist'
    param['silent'] = 1

    print("Loading data ...")
    dtrain, dtest, train_label, test_label = get_data(fpath)

    print("Training with GPU ...")
    tmp = time.time()
    model = xgb.train(param, dtrain, num_round)
    train_time = time.time() - tmp
    print("GPU Training Time: %s seconds" % (str(train_time)))

    print("Testing with GPU ...")
    spred = time.time()
    predictions = model.predict(dtest)
    test_time = time.time() - spred

    mean_acc = calc_metric(predictions, test_label)

    print('mean_acc: ',mean_acc)
    print("GPU Testing Time: %s seconds" % (str(test_time)))

def train_CPU():

    param = {}
    param['objective'] = 'binary:logitraw'
    param['eval_metric'] = 'error'
    param['tree_method'] = 'hist'
    # param['silent'] = 1

    print("Loading data ...")
    dtrain, dtest, train_label, test_label = get_data(fpath)
    tmp = time.time()

    print("Training with CPU ...")
    model = xgb.train(param, dtrain, num_round)
    train_time = time.time() - tmp
    print("CPU Training Time: %s seconds" % (str(train_time)))

    print("Testing with CPU ...")
    spred = time.time()
    predictions = model.predict(dtest)
    test_time = time.time() - spred

    mean_acc = calc_metric(predictions, test_label)
    print('mean_acc: ',mean_acc)
    print("CPU Testing Time: %s seconds" % (str(test_time)))

if __name__ == "__main__":

    train_rows = 1050000
    test_rows = 50000
    num_round = 1000        

    fpath = '../data/HIGGS.csv'

    # train_GPU()

    train_CPU()