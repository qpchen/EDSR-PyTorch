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
# for extremly tiny size (XT) inf:K

export CUDA_VISIBLE_DEVICES=3
nohup python main.py --n_GPUs 1 --scale 4 --patch_size 192 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --decay 1500-2400-2700-2850 --res_connect 1acb3 --srarn_up_feat 24 --depths 2+2+2+2 --dims 24+24+24+24 --model SRARNV6 --save ../srarn_v6/srarn_v6xt_j15_x4 --reset > ../srarn_v6/v6xt_j15_x4.log 2>&1 &

