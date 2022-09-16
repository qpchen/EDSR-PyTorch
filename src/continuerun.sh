#!/bin/bash

# set python output immediately
export PYTHONUNBUFFERED=1


export CUDA_VISIBLE_DEVICES=1
nohup python main.py --n_GPUs 1 --model AAN --scale 4 --patch_size 256 --batch_size 32 --lr 5e-4 --epochs 2000 --skip_threshold 1e6 --data_test Set5 --gclip 5 --load ann_gc_x4 --resume -1 > train_ann_x4.out 2>&1 &
export CUDA_VISIBLE_DEVICES=1
nohup python main.py --n_GPUs 1 --scale 2 --patch_size 128 --lr 5e-4 --epochs 2000 --skip_threshold 1e6 --data_test Set5 --batch_size 32 --gclip 5 --model BIAANV3H --load biann_v3hg_x2 --resume -1 > train_v3hg.out 2>&1 &
export CUDA_VISIBLE_DEVICES=2
nohup python main.py --n_GPUs 1 --scale 2 --patch_size 128 --lr 5e-4 --epochs 2000 --skip_threshold 1e6 --data_test Set5 --batch_size 32 --gclip 5 --model BIAANV12 --reset --save biann_v12_x2 > train_v12.out 2>&1 &
export CUDA_VISIBLE_DEVICES=3
nohup python main.py --n_GPUs 1 --scale 2 --patch_size 128 --lr 5e-4 --epochs 2000 --skip_threshold 1e6 --data_test Set5 --batch_size 32 --gclip 5 --model BIAANV9C --load biann_nov9cg_x2 --resume -1 > train_nov9cg.out 2>&1 &

#####################################################################
# can use '--jobid=id' to continue yhbatch/yhrun jobs to old queue
#####################################################################
