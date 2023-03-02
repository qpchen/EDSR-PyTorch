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
# like SwinIR-S
export CUDA_VISIBLE_DEVICES=0
nohup python main.py --n_GPUs 1 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 1500 --decay 750-1200-1350-1425 --res_connect 1acb3 --srarn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --model SRARNV6 --save ../srarn_v6/srarn_v6s_f11_x2 --reset > ../srarn_v6/v6s_f11_x2.log 2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python main.py --n_GPUs 1 --scale 3 --patch_size 144 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 1500 --decay 750-1200-1350-1425 --res_connect 1acb3 --srarn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --model SRARNV6 --save ../srarn_v6/srarn_v6s_f11_x3 --reset > ../srarn_v6/v6s_f11_x3.log 2>&1 &
