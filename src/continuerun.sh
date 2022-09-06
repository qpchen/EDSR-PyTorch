#!/bin/bash

# set python output immediately
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
nohup python main.py --n_GPUs 1 --scale 2 --patch_size 128 --lr 1e-4 --epochs 2000 --skip_threshold 1e6 --data_test Set5 --batch_size 32 --gclip 5 --model BIAANV9B --reset --save biann_v9bg_x2  > train_v3g.out 2>&1 &
export CUDA_VISIBLE_DEVICES=2
nohup python main.py --n_GPUs 1 --scale 2 --patch_size 128 --lr 1e-4 --epochs 2000 --skip_threshold 1e6 --data_test Set5 --batch_size 32 --model BIAANV9C --reset --save biann_v9c_x2  > train_v3f.out 2>&1 &

#export CUDA_VISIBLE_DEVICES=0
#nohup python main.py --n_GPUs 1 --scale 2 --patch_size 128 --lr 1e-4 --epochs 2000 --skip_threshold 1e6 --data_test Set5 --batch_size 32 --model BIAANV9A --reset --save biann_v9a_x2  > train_v9a.out 2>&1 &
export CUDA_VISIBLE_DEVICES=3
nohup python main.py --n_GPUs 1 --model AAN --scale 4 --patch_size 256 --batch_size 32  --lr 5e-4 --epochs 2000 --skip_threshold 1e6 --data_test Set5 --gclip 5 --reset --save ann_gc_x4 > train_ann_x4.out 2>&1 &
