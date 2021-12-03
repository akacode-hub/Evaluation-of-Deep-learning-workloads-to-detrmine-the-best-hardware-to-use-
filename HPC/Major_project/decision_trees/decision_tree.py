import csv
import numpy as np
import os.path
import pandas
import time
import xgboost as xgb
import sys


def train():

    dtrain, dtest = load_higgs()
    param = {}
    param['objective'] = 'binary:logitraw'
    param['eval_metric'] = 'error'
    param['tree_method'] = 'gpu_hist'
    param['silent'] = 1

    print("Training with GPU ...")
    tmp = time.time()
    gpu_res = {}
    xgb.train(param, dtrain, num_round, evals=[(dtest, "test")], 
            evals_result=gpu_res)
    gpu_time = time.time() - tmp
    print("GPU Training Time: %s seconds" % (str(gpu_time)))

    print("Training with CPU ...")
    param['tree_method'] = 'hist'
    tmp = time.time()
    cpu_res = {}
    xgb.train(param, dtrain, num_round, evals=[(dtest, "test")], 
            evals_result=cpu_res)
    cpu_time = time.time() - tmp
    print("CPU Training Time: %s seconds" % (str(cpu_time)))

    if plot:
        import matplotlib.pyplot as plt
        min_error = min(min(gpu_res["test"][param['eval_metric']]), 
                        min(cpu_res["test"][param['eval_metric']]))
        gpu_iteration_time = 
            [x / (num_round * 1.0) * gpu_time for x in range(0, num_round)]
        cpu_iteration_time = 
            [x / (num_round * 1.0) * cpu_time for x in range(0, num_round)]
        plt.plot(gpu_iteration_time, gpu_res['test'][param['eval_metric']], 
                label='Tesla P100')
        plt.plot(cpu_iteration_time, cpu_res['test'][param['eval_metric']], 
                label='2x Haswell E5-2698 v3 (32 cores)')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Test error')
        plt.axhline(y=min_error, color='r', linestyle='dashed')
        plt.margins(x=0)
        plt.ylim((0.23, 0.35))
        plt.show()