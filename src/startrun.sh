#!/bin/bash

#########################################
# run this script use command following:
# yhbatch -N 1-1 -n 1 -p gpu_v100 startrun.sh
#########################################

#export CUDA_VISIBLE_DEVICES=2,3
#yhrun -n 1 -N 1-1 -p gpu_v100 python main.py --n_GPUs 2 --model AAN --scale 4 --patch_size 256 --batch_size 32  --lr 5e-4 --epochs 2000 --skip_threshold 1e6 --data_test Set5 --reset --save ann_x4


#export CUDA_VISIBLE_DEVICES=0
#yhrun -n 1 -N 1-1 -p gpu_v100 python main.py --n_GPUs 1 --scale 2 --patch_size 128 --lr 5e-4 --epochs 2000 --skip_threshold 1e6 --data_test Set5 --batch_size 32 --gclip 5 --model BIAANV11 --reset --save biann_v11g_x2

yhrun -n 1 -N 1-1 -p gpu_v100 python main.py --n_GPUs 1 --scale 2 --patch_size 128 --batch_size 32 --data_test Set5 --data_range 1-900 --loss 1*MSE --lr 1e-3 --n_colors 1 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --model BIFSRCNNPSV3 --reset --save bifsrcnnps_v3a_ls_x2

#python main.py --n_GPUs 1 --scale 2 --patch_size 128 --batch_size 32 --data_test Set5 --data_range 1-900 --loss 1*MSE --lr 1e-3 --n_colors 1 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --model BIFSRCNNLIV4 --reset --save bifsrcnnLI_v4_x2