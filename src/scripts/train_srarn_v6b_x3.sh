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
# like SwinIR-B

export CUDA_VISIBLE_DEVICES=0,1
nohup python main.py --n_GPUs 2 --scale 3 --patch_size 144 --batch_size 32 --accumulation_step 2 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 1000 --decay 500-800-900-950 --res_connect 1acb3 --upsampling PixelShuffle --srarn_up_feat 64 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --model SRARNV6 --save ../srarn_v6/srarn_v6_ps_g12_x3 --reset > ../srarn_v6/v6_ps_g12_x3.log 2>&1 &
