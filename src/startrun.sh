#!/bin/bash

#########################################
# run this script use command following:
# yhbatch -N 1-1 -n 1 -p gpu_v100 startrun.sh
#########################################

#export CUDA_VISIBLE_DEVICES=2,3
#yhrun -n 1 -N 1-1 -p gpu_v100 python main.py --n_GPUs 2 --model AAN --scale 4 --patch_size 256 --batch_size 32  --lr 5e-4 --epochs 2000 --skip_threshold 1e6 --data_test Set5 --reset --save ann_x4


#export CUDA_VISIBLE_DEVICES=0
#yhrun -n 1 -N 1-1 -p gpu_v100 python main.py --n_GPUs 1 --scale 2 --patch_size 128 --lr 5e-4 --epochs 2000 --skip_threshold 1e6 --data_test Set5 --batch_size 32 --gclip 5 --model BIAANV11 --reset --save biann_v11g_x2

# yhrun -n 1 -N 1-1 -p gpu_v100 python main.py --n_GPUs 1 --scale 2 --patch_size 128 --batch_size 32 --data_test Set5 --data_range 1-900 --loss 1*MSE --lr 1e-3 --n_colors 1 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --model BIFSRCNNPSV3 --reset --save bifsrcnnps_v3a_ls_x2

#python main.py --n_GPUs 1 --scale 2 --patch_size 128 --batch_size 32 --data_test Set5 --data_range 1-900 --loss 1*MSE --lr 1e-3 --n_colors 1 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --model BIFSRCNNLIV4 --reset --save bifsrcnnLI_v4_x2

# ###################################
# SRARN settings like ConvNeXt
# yhrun -n 1 -N 1-1 -p gpu_v100 python main.py --n_GPUs 4 --scale 2 --patch_size 128 --batch_size 32 --data_test Set5 --loss 1*SmoothL1 --lr 1e-3 --n_colors 1 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --depths 3+3+27+3 --dims 128+256+512+1024 --model SRARNV2 --save ../srarn/srarn_v2e7_x2 --reset

# yhrun -n 1 -N 1-1 -p gpu_v100 python main.py --n_GPUs 2 --scale 2 --patch_size 128 --batch_size 32 --skip_threshold 1e6 --epochs 3000 --data_test Set5+Set14 --loss 1*SmoothL1 --lr 4e-3 --n_colors 1 --optimizer AdamW --weight_decay 0.05 --depths 3+3+9+3 --dims 48+96+192+384 --model SRARNV2 --save ../srarn/srarn_v2d5_div_opt_x2 --reset

# yhrun -n 1 -N 1-1 -p gpu_v100 python main.py --n_GPUs 2 --scale 2 --patch_size 128 --batch_size 32 --skip_threshold 1e6 --epochs 3000 --data_train DF2K --data_range 1-3550 --data_test Set5+Set14 --loss 1*SmoothL1 --lr 4e-3 --n_colors 1 --optimizer AdamW --weight_decay 0.05 --depths 3+3+9+3 --dims 48+96+192+384 --model SRARNV2 --save ../srarn/srarn_v2d5_df_opt_x2 --reset

# 得分比较高的设置
# nohup python main.py --n_GPUs 2 --scale 2 --patch_size 128 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 1e-3 --n_colors 1 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --n_up_feat 64 --depths 3+3+9+3 --dims 48+96+192+384 --model SRARNV2 --save ../srarn/srarn_v2d5_x2 --reset > ../srarn/v2d5.log 2>&1 &


# nohup python main.py --n_GPUs 2 --scale 2 --patch_size 128 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 1e-3 --n_colors 1 --optimizer ADAM --skip_threshold 1e6 --epochs 3000  --srarn_up_feat 64 --depths 3+3+9+3 --dims 48+96+192+384 --model SRARNV3 --save ../srarn/srarn_v3d5_x2 --reset > ../srarn/v3d5_x2.log 2>&1 &

# nohup python main.py --n_GPUs 4 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 1 --optimizer ADAM --skip_threshold 1e6 --epochs 3000  --srarn_up_feat 0 --depths 3+3+27+3 --dims 128+256+512+1024 --model SRARNV3 --save ../srarn/srarn_v3e8_x2 --reset > ../srarn/v3e8_x2.log 2>&1 &

# #####################################
# SRARN settings like SwinIR, but the params is too large, which is 60M
# nohup python main.py --n_GPUs 4 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000  --srarn_up_feat 64 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --model SRARNV3 --save ../srarn/srarn_v3g12_x2 --reset > ../srarn/v3g12_x2.log 2>&1 &

