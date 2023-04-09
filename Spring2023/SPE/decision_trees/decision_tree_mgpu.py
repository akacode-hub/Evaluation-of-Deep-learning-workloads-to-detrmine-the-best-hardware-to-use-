import csv
import numpy as np
import os.path
import pandas as pd
import time
import xgboost as xgb
import sys
import dask.dataframe as dd
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from xgboost.dask import DaskDMatrix

def get_data(filename):
    
    higgs_train = pd.read_csv(filename, dtype=np.float32, 
                                     nrows=train_rows, header=None)

    higgs_test = pd.read_csv(filename, dtype=np.float32, 
                                    skiprows=train_rows, nrows=test_rows, 
                                    header=None)

    train_label = higgs_train.iloc[:, 0].values
    test_label = higgs_test.iloc[:, 0].values

    higgs_train = dd.from_pandas(higgs_train, npartitions=8)
    higgs_test = dd.from_pandas(higgs_test, npartitions=8)

    return higgs_train, higgs_test, train_label, test_label

def calc_metric(output, label, thresh=0.5):

    output = np.asarray(output)
    output = (output>thresh).astype('int')

    return 1 - np.mean(np.abs(output - label))

def train(client):

    param = {}
    param['objective'] = 'binary:logitraw'
    param['eval_metric'] = 'error'
    param['tree_method'] = 'gpu_hist'
    param['nthread'] = 1
    param['silence'] = 1

    print("Loading data ...")
    dtrain, dtest, train_label, test_label = get_data(fpath)

    dtrain = DaskDMatrix(client, dtrain.loc[:, 1:29], dtrain[0])
    dtest = DaskDMatrix(client, dtest.loc[:, 1:29], dtest[0])

    print("Training with GPU ...")
    tmp = time.time()
    output = xgb.dask.train(client, param, dtrain, num_boost_round=num_boost_round, evals=[])
    gpu_time = time.time() - tmp
    print("GPU Training Time: %s seconds" % (str(gpu_time)))

    bst = output['booster']
    history = output['history']

    print("Testing with GPU ...")
    spred = time.time()
    predictions = xgb.dask.predict(client, bst, dtest)
    test_time = time.time() - spred

    mean_acc = calc_metric(predictions, test_label)

    print('mean_acc: ',mean_acc)
    print("GPU Testing Time: %s seconds" % (str(test_time)))

if __name__ == "__main__":

    train_rows = 1050000
    test_rows = 50000
    num_boost_round = 1000

    fpath = '../data/HIGGS.csv'
    num_gpus = 4

    # `LocalCUDACluster` is used for assigning GPU to XGBoost processes.  Here
    # `n_workers` represents the number of GPUs since we use one GPU per worker
    # process.
    with LocalCUDACluster(n_workers=num_gpus, threads_per_worker=4) as cluster:
        with Client(cluster) as client:
            train(client)
