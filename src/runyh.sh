#!/bin/bash

yhrun -n 1 -N 1-1 -p gpu_v100 CUDA_VISIBLE_DEVICES=2,3 python main.py --n_GPUs 2 --scale 2 --patch_size 128 --lr 1e-4 --epochs 2000 --skip_threshold 1e6 --data_test Set5 --batch_size 32 --model BIAANV3F --reset --save biann_v3f_x2
yhrun -n 1 -N 1-1 -p gpu_v100 CUDA_VISIBLE_DEVICES=0,1 python main.py --n_GPUs 2 --scale 2 --patch_size 128 --lr 1e-4 --epochs 2000 --skip_threshold 1e6 --data_test Set5 --batch_size 32 --model BIAANV9A --reset --save biann_v9a_x2