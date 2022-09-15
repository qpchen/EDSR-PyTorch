#!/bin/bash

#########################################
# run this script use command following:
# yhbatch -N 1-1 -n 1 -p gpu_v100 startrun.sh
#########################################

#export CUDA_VISIBLE_DEVICES=2,3
#yhrun -n 1 -N 1-1 -p gpu_v100 python main.py --n_GPUs 2 --model AAN --scale 4 --patch_size 256 --batch_size 32  --lr 5e-4 --epochs 2000 --skip_threshold 1e6 --data_test Set5 --reset --save ann_x4


export CUDA_VISIBLE_DEVICES=0
yhrun -n 1 -N 1-1 -p gpu_v100 python main.py --n_GPUs 1 --scale 2 --patch_size 128 --lr 5e-4 --epochs 2000 --skip_threshold 1e6 --data_test Set5 --batch_size 32 --gclip 5 --model BIAANV11 --reset --save biann_v11g_x2
