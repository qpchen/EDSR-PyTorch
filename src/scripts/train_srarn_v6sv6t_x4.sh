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
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# nohup python main.py --n_GPUs 4 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 1000 --decay 500-800-900-950 --res_connect 1acb3 --upsampling PixelShuffle --srarn_up_feat 64 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --model SRARNV6 --save ../srarn_v6/srarn_v6_ps_g12_x2 --reset > ../srarn_v6/v6_ps_g12_x2.log 2>&1 &
# Test:
# python main.py --n_GPUs 4 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 1000 --res_connect 1acb3 --upsampling PixelShuffle --srarn_up_feat 64 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --model SRARNV6 --save ../srarn_v6/srarn_v6_ps_g12_x2 --pre_train ../srarn_v6/srarn_v6_ps_g12_x2/model/model_best.pt --test_only --save_result --inf_switch
# python main.py --n_GPUs 4 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 1000 --res_connect 1acb3 --upsampling PixelShuffle --srarn_up_feat 64 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --model SRARNV6 --save ../srarn_v6/srarn_v6_ps_g12_x2 --pre_train ../srarn_v6/srarn_v6_ps_g12_x2/model/inf_model.pt --test_only --save_result --load_inf

# export CUDA_VISIBLE_DEVICES=0,1
# nohup python main.py --n_GPUs 2 --scale 3 --patch_size 144 --batch_size 32 --accumulation_step 2 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 1000 --decay 500-800-900-950 --res_connect 1acb3 --upsampling PixelShuffle --srarn_up_feat 64 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --model SRARNV6 --save ../srarn_v6/srarn_v6_ps_g12_x3 --reset > ../srarn_v6/v6_ps_g12_x3.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=0,1
# nohup python main.py --n_GPUs 2 --scale 4 --patch_size 192 --batch_size 32 --accumulation_step 2 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 1000 --decay 500-800-900-950 --res_connect 1acb3 --upsampling PixelShuffle --srarn_up_feat 64 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --model SRARNV6 --save ../srarn_v6/srarn_v6_ps_g12_x4 --reset > ../srarn_v6/v6_ps_g12_x4.log 2>&1 &

# #####################################
# like SwinIR-S
# export CUDA_VISIBLE_DEVICES=0
# nohup python main.py --n_GPUs 1 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 1500 --decay 750-1200-1350-1425 --res_connect 1acb3 --srarn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --model SRARNV6 --save ../srarn_v6/srarn_v6s_f11_x2 --reset > ../srarn_v6/v6s_f11_x2.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=0
# nohup python main.py --n_GPUs 1 --scale 3 --patch_size 144 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 1500 --decay 750-1200-1350-1425 --res_connect 1acb3 --srarn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --model SRARNV6 --save ../srarn_v6/srarn_v6s_f11_x3 --reset > ../srarn_v6/v6s_f11_x3.log 2>&1 &

export CUDA_VISIBLE_DEVICES=0
nohup python main.py --n_GPUs 1 --scale 4 --patch_size 192 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 1500 --decay 750-1200-1350-1425 --res_connect 1acb3 --srarn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --model SRARNV6 --save ../srarn_v6/srarn_v6s_f11_x4 --reset > ../srarn_v6/v6s_f11_x4.log 2>&1 &


# #####################################
# for tiny size (T) c14
# export CUDA_VISIBLE_DEVICES=0
# nohup python main.py --n_GPUs 1 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 2000 --decay 1000-1600-1800-1900 --res_connect 1acb3 --srarn_up_feat 30 --depths 2+2+6+2 --dims 30+30+30+30 --model SRARNV6 --save ../srarn_v6/srarn_v6t_c14_x2 --reset > ../srarn_v6/v6t_c14_x2.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=0
# nohup python main.py --n_GPUs 1 --scale 3 --patch_size 144 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 2000 --decay 1000-1600-1800-1900 --res_connect 1acb3 --srarn_up_feat 30 --depths 2+2+6+2 --dims 30+30+30+30 --model SRARNV6 --save ../srarn_v6/srarn_v6t_c14_x3 --reset > ../srarn_v6/v6t_c14_x3.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=0
# nohup python main.py --n_GPUs 1 --scale 4 --patch_size 192 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 2000 --decay 1000-1600-1800-1900 --res_connect 1acb3 --srarn_up_feat 30 --depths 2+2+6+2 --dims 30+30+30+30 --model SRARNV6 --save ../srarn_v6/srarn_v6t_c14_x4 --reset > ../srarn_v6/v6t_c14_x4.log 2>&1 &


# for tiny size (T) i14
# export CUDA_VISIBLE_DEVICES=0
# nohup python main.py --n_GPUs 1 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 2000 --decay 1000-1600-1800-1900 --res_connect 1acb3 --srarn_up_feat 30 --depths 3+3+3+3 --dims 30+30+30+30 --model SRARNV6 --save ../srarn_v6/srarn_v6t_i14_x2 --reset > ../srarn_v6/v6t_i14_x2.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=0
# nohup python main.py --n_GPUs 1 --scale 3 --patch_size 144 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 2000 --decay 1000-1600-1800-1900 --res_connect 1acb3 --srarn_up_feat 30 --depths 3+3+3+3 --dims 30+30+30+30 --model SRARNV6 --save ../srarn_v6/srarn_v6t_i14_x3 --reset > ../srarn_v6/v6t_i14_x3.log 2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python main.py --n_GPUs 1 --scale 4 --patch_size 192 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 2000 --decay 1000-1600-1800-1900 --res_connect 1acb3 --srarn_up_feat 30 --depths 3+3+3+3 --dims 30+30+30+30 --model SRARNV6 --save ../srarn_v6/srarn_v6t_i14_x4 --reset > ../srarn_v6/v6t_i14_x4.log 2>&1 &


# #####################################
# for extremly tiny size (XT) inf:K
# export CUDA_VISIBLE_DEVICES=0
# nohup python main.py --n_GPUs 1 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --decay 1500-2400-2700-2850 --decay 2750 --res_connect 1acb3 --srarn_up_feat 24 --depths 2+2+2+2 --dims 24+24+24+24 --model SRARNV6 --save ../srarn_v6/srarn_v6xt_j15_x2 --reset > ../srarn_v6/v6xt_j15_x2.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=0
# nohup python main.py --n_GPUs 1 --scale 3 --patch_size 144 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --decay 1500-2400-2700-2850 --res_connect 1acb3 --srarn_up_feat 24 --depths 2+2+2+2 --dims 24+24+24+24 --model SRARNV6 --save ../srarn_v6/srarn_v6xt_j15_x3 --reset > ../srarn_v6/v6xt_j15_x3.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=0
# nohup python main.py --n_GPUs 1 --scale 4 --patch_size 192 --batch_size 32 --data_test Set5 --loss 1\*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --epochs 3000 --decay 1500-2400-2700-2850 --res_connect 1acb3 --srarn_up_feat 24 --depths 2+2+2+2 --dims 24+24+24+24 --model SRARNV6 --save ../srarn_v6/srarn_v6xt_j15_x4 --reset > ../srarn_v6/v6xt_j15_x4.log 2>&1 &

