import csv
import numpy as np
import os.path
import pandas as pd
import time
import xgboost as xgb
import sys

def get_data(filename):
    
    higgs_train = pd.read_csv(filename, dtype=np.float32, 
                                     nrows=train_rows, header=None)

    higgs_test = pd.read_csv(filename, dtype=np.float32, 
                                    skiprows=train_rows, nrows=test_rows, 
                                    header=None)

    higgs_train = xgb.DMatrix(higgs_train.ix[:, 1:29], higgs_train[0])
    higgs_test = xgb.DMatrix(higgs_test.ix[:, 1:29], higgs_test[0])

    return higgs_train, higgs_test

def train_GPU():

    param = {}
    param['objective'] = 'binary:logitraw'
    param['eval_metric'] = 'error'
    param['tree_method'] = 'gpu_hist'
    param['silent'] = 1

    print("Loading data ...")
    dtrain, dtest = get_data(fpath)
    tmp = time.time()
    gpu_res = {}
    print("Training with GPU ...")
    xgb.train(param, dtrain, num_round, evals=[(dtest, "test")], 
            evals_result=gpu_res)
    gpu_time = time.time() - tmp
    print("GPU Training Time: %s seconds" % (str(gpu_time)))

def train_CPU():

    param = {}
    param['objective'] = 'binary:logitraw'
    param['eval_metric'] = 'error'
    param['tree_method'] = 'hist'
    param['silent'] = 1

    print("Training with CPU ...")
    dtrain, dtest = get_data(fpath)
    tmp = time.time()
    cpu_res = {}
    xgb.train(param, dtrain, num_round, evals=[(dtest, "test")], 
            evals_result=cpu_res)
    cpu_time = time.time() - tmp
    print("CPU Training Time: %s seconds" % (str(cpu_time)))

if __name__ == "__main__":

    train_rows = 10500000
    test_rows = 500000
    num_round = 1000        

    fpath = '../dataset/HIGGS.csv'

    train_GPU()

    train_CPU()