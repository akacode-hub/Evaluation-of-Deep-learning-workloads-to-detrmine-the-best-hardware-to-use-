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

def get_data(client: Client, filename):
    
    higgs_train = pd.read_csv(filename, dtype=np.float32, 
                                     nrows=train_rows, header=None)

    higgs_test = pd.read_csv(filename, dtype=np.float32, 
                                    skiprows=train_rows, nrows=test_rows, 
                                    header=None)

    higgs_train = DaskDMatrix(higgs_train.ix[:, 1:29], higgs_train[0])
    higgs_train = DaskDMatrix(higgs_test.ix[:, 1:29], higgs_test[0])

    return higgs_train, higgs_test

def train(client: Client):

    param = {}
    param['objective'] = 'binary:logitraw'
    param['eval_metric'] = 'error'
    param['tree_method'] = 'gpu_hist'
    param['silent'] = 1

    print("Training with GPU ...")
    dtrain, dtest = get_data(fpath)
    tmp = time.time()
    gpu_res = {}
    xgb.train(param, dtrain, num_round, evals=[(dtest, "test")], 
            evals_result=gpu_res)
    gpu_time = time.time() - tmp
    print("GPU Training Time: %s seconds" % (str(gpu_time)))

if __name__ == "__main__":

    train_rows = 10500000
    test_rows = 500000
    num_round = 1000        

    fpath = '../dataset/HIGGS.csv'
    num_gpus = 2
    # `LocalCUDACluster` is used for assigning GPU to XGBoost processes.  Here
    # `n_workers` represents the number of GPUs since we use one GPU per worker
    # process.
    with LocalCUDACluster(n_workers=num_gpus, threads_per_worker=4) as cluster:
        with Client(cluster) as client:
            train(client)
