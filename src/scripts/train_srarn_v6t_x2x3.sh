#!/bin/bash

#########################################
# run this script at starlight use command following:
# yhbatch -N 1-1 -n 1 -p gpu_v100 startrun.sh
#########################################

# yhrun -n 1 -N 1-1 -p gpu_v100 python main.py --n_GPUs 1 --scale 2 --patch_size 128 --batch_size 32 --data_test Set5 --data_range 1-900 --loss 1*MSE --lr 1e-3 --n_colors 1 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --model BIFSRCNNPSV3 --reset --save bifsrcnnps_v3a_ls_x2

################################################################################
######################      SRARN V6       ######################
################################################################################

# #####################################
# for tiny size (T) i14
export CUDA_VISIBLE_DEVICES=0
nohup python main.py --n_GPUs 1 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 2000 --decay 1000-1600-1800-1900 --res_connect 1acb3 --srarn_up_feat 30 --depths 3+3+3+3 --dims 30+30+30+30 --model SRARNV6 --save ../srarn_v6/srarn_v6t_i14_x2 --reset > ../srarn_v6/v6t_i14_x2.log 2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python main.py --n_GPUs 1 --scale 3 --patch_size 144 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 2000 --decay 1000-1600-1800-1900 --res_connect 1acb3 --srarn_up_feat 30 --depths 3+3+3+3 --dims 30+30+30+30 --model SRARNV6 --save ../srarn_v6/srarn_v6t_i14_x3 --reset > ../srarn_v6/v6t_i14_x3.log 2>&1 &

