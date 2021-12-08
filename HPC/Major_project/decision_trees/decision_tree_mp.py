import csv
import numpy as np
import os.path
import pandas as pd
import time
import xgboost as xgb
import sys
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from xgboost.dask import DaskDMatrix

def get_data(filename):
    
    higgs_train = pd.read_csv(filename, dtype=np.float32, 
                                     nrows=train_rows, header=None)

    higgs_test = pd.read_csv(filename, dtype=np.float32, 
                                    skiprows=train_rows, nrows=test_rows, 
                                    header=None)

    return higgs_train, higgs_test

def train(client):

    param = {}
    param['objective'] = 'binary:logitraw'
    param['eval_metric'] = 'error'
    param['tree_method'] = 'gpu_hist'
    param['silent'] = 1

    print("Loading data ...")
    dtrain, dtest = get_data(fpath)
    dtrain = DaskDMatrix(client, dtrain.loc[:, 1:29], dtrain[0])
    dtest = DaskDMatrix(client, dtest.loc[:, 1:29], dtest[0])

    tmp = time.time()
    gpu_res = {}
    print("Training with GPU ...")
    output = xgb.dask.train(client, param, dtrain, num_boost_round=4, evals=[(dtest, "test")], 
            evals_result=gpu_res)
    gpu_time = time.time() - tmp
    print("GPU Training Time: %s seconds" % (str(gpu_time)))

if __name__ == "__main__":

    train_rows = 1050#0000
    test_rows = 500#000
    num_round = 1000        

    fpath = '../dataset/HIGGS.csv'
    num_gpus = 3

    # `LocalCUDACluster` is used for assigning GPU to XGBoost processes.  Here
    # `n_workers` represents the number of GPUs since we use one GPU per worker
    # process.
    with LocalCUDACluster(n_workers=num_gpus, threads_per_worker=4) as cluster:
        with Client(cluster) as client:
            train(client)
