## Decision Trees

This repository contains the following script:
* decision_tree.py - Decision Trees training for CPU and GPU.
* decision_tree_mgpu.py - Decision Trees training for multi-GPU 2 and 4.

## Run instructions

* To run this program, run the below command in a bash script. The bash script used `train.sh` is provided.

* To train for CPU,

    `CUDA_VISIBLE_DEVICES="-1" python3 decision_tree.py`

* To train for GPU,

    `CUDA_VISIBLE_DEVICES="0" python3 decision_tree.py`

* To train for multi-GPU(2),

    `CUDA_VISIBLE_DEVICES="0, 1" python3 decision_tree_mgpu.py`

* To train for multi-GPU(4),

    `CUDA_VISIBLE_DEVICES="0, 1, 2, 3" python3 decision_tree_mgpu.py`

## Log
* `logs/exp_v1.log` provides sample log for training decision trees on CPU and GPU.
* `logs/exp_v2.log` provides sample log for training decision trees on multi-GPU(2).
* `logs/exp_v3.log` provides sample log for training decision trees on multi-GPU(4).