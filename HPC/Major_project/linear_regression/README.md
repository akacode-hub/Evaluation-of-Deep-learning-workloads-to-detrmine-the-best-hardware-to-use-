## Linear Regression

This repository contains the following script:
* lin_reg.py - Linear Regression training for CPU and GPU.
* lin_reg_mgpu.py - Linear Regression training for multi-GPU 2 and 4.

## Run instructions

* To run this program, run the below command in a bash script. The bash script used `train.sh` is provided.

* To train for CPU,

    `CUDA_VISIBLE_DEVICES="-1" python3 lin_reg.py`

* To train for GPU,

    `CUDA_VISIBLE_DEVICES="0" python3 lin_reg.py`

* To train for multi-GPU(2),

    `CUDA_VISIBLE_DEVICES="0, 1" python3 lin_reg_mgpu.py`

* To train for multi-GPU(4),

    `CUDA_VISIBLE_DEVICES="0, 1, 2, 3" python3 lin_reg_mgpu.py`

## Log
* `logs/exp_v1.log` provides sample log for training linear regression model on CPU.
* `logs/exp_v2.log` provides sample log for training linear regression model on GPU.
* `logs/exp_v3.log` provides sample log for training linear regression model on multi-GPU(2).
* `logs/exp_v4.log` provides sample log for training linear regression model on multi-GPU(4).
