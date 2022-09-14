#!/bin/bash

# set python output immediately
export PYTHONUNBUFFERED=1

export CUDA_VISIBLE_DEVICES=1
nohup python main.py --n_GPUs 1 --scale 2 --patch_size 128 --lr 5e-4 --epochs 2000 --skip_threshold 1e6 --data_test Set5 --batch_size 32 --model BIAANV9A --reset --save biann_nov9a_x2 > train_nov9a.out 2>&1 &
export CUDA_VISIBLE_DEVICES=2
nohup python main.py --n_GPUs 1 --scale 2 --patch_size 128 --lr 5e-4 --epochs 2000 --skip_threshold 1e6 --data_test Set5 --batch_size 32 --model BIAANV9B --reset --save biann_nov9b_x2 > train_nov9b.out 2>&1 &
export CUDA_VISIBLE_DEVICES=3
nohup python main.py --n_GPUs 1 --scale 2 --patch_size 128 --lr 5e-4 --epochs 2000 --skip_threshold 1e6 --data_test Set5 --batch_size 32 --model BIAANV9C --reset --save biann_nov9c_x2 > train_nov9c.out 2>&1 &

#####################################################################
# can use '--jobid=id' to continue yhrun jobs to old queue
#####################################################################

# cannot use yhbatch to run this script, may need some help
#startid=999
#read -rp "Enter exist jobid, please: " startid
#export CUDA_VISIBLE_DEVICES=1
#yhrun -n 1 -N 1-1 -p gpu_v100 --jobid=$startid python main.py --n_GPUs 1 --scale 2 --patch_size 128 --lr 5e-4 --epochs 2000 --skip_threshold 1e6 --data_test Set5 --batch_size 32 --model BIAANV9A --reset --save biann_nov9a_x2
#export CUDA_VISIBLE_DEVICES=2
#yhrun -n 1 -N 1-1 -p gpu_v100 --jobid=$startid python main.py --n_GPUs 1 --scale 2 --patch_size 128 --lr 5e-4 --epochs 2000 --skip_threshold 1e6 --data_test Set5 --batch_size 32 --model BIAANV9B --reset --save biann_nov9b_x2
#export CUDA_VISIBLE_DEVICES=3
#yhrun -n 1 -N 1-1 -p gpu_v100 --jobid=$startid python main.py --n_GPUs 1 --scale 2 --patch_size 128 --lr 5e-4 --epochs 2000 --skip_threshold 1e6 --data_test Set5 --batch_size 32 --model BIAANV9C --reset --save biann_nov9c_x2
