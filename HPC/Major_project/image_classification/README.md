## Image Classification

This repository contains the following script:
* train.py - Image Classification training for GPU - Single Precision.
* train_mgpu.py - Image Classification training for multi-GPU 2 and 4 - Single Precision(FP32).
* train_amp.py - Image Classification training for GPUs 1, 2 and 4 - Mixed Precision(FP16).

## Run instructions

* To train for GPU - FP32,

    `CUDA_VISIBLE_DEVICES="0" python3 train.py`

* To train for multi-GPU(4) - FP32,

    `CUDA_VISIBLE_DEVICES="0, 1, 2, 3" python3 train_mgpu.py`

* To train for multi-GPU(4) - FP16,

    `CUDA_VISIBLE_DEVICES="0, 1, 2, 3" python3 train_amp.py`