#!/bin/bash

#########################################
# This script is used for test the runtime of single 720P image
# run this script use command following (if just have one size, just input anything):
# ./test_runtime.sh [model_name] [size] [scale] [patch] [times] "[addition options]"
# #### specially for SRARN model ,just input Version, and need addition model size param:
# #### ./test_runtime.sh [version] [size] [scale] [patch] [times] "[addition options]"
#########################################
# after copy ../runtime_models/*.pt files and ../../dataset/runtime dir, run example like: 
# ./scripts/test_runtime.sh v5 xt 2 48 10 "--no_count --save_result"
#########################################


# python main.py --n_GPUs 1 --accumulation_step 1 --scale 2 --patch_size 96 --epochs 3000 --decay 1500-2400-2700-2850 --upsampling Nearest --srarn_up_feat 24 --depths 2+2+2+2 --dims 24+24+24+24 --batch_size 32  --res_connect skip --acb_norm v8old --loss 1*SmoothL1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --lr_class MultiStepLR --model SRARNV8 --data_test Set5+Set14+B100+Urban100+Manga109 --save ../srarn_v8/v8xt_x2_test --pre_train ../srarn_v8/v8xt_x2_test/model/inf_model.pt --test_only --save_result --load_inf

# python test_runtime.py --n_threads 1 --cpu --scale 2 --patch_size 96 --batch_size 32 --data_test 720P --n_colors 3 --res_connect 1acb3 --srarn_up_feat 24 --depths 2+2+2+2 --dims 24+24+24+24 --model SRARNV5 --save ../runtime_models/logs/v5xt_x2 --pre_train ../runtime_models/v5xt_x2.pt --test_only --load_inf --reset --runtime --times 10 --no_count --save_result

# FSRCNN
model=$1
# patch=`expr $3 \* 48`
patch=`expr $3 \* $4`
if [ $2 = "xt" ]; then
  dim_configs="--srarn_up_feat 24 --depths 2+2+2+2 --dims 24+24+24+24"
elif [ $2 = "t" ]; then
  dim_configs="--srarn_up_feat 30 --depths 3+3+3+3 --dims 30+30+30+30"
elif [ $2 = "s" ]; then
  dim_configs="--srarn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60"
elif [ $2 = "b" ]; then
  dim_configs="--srarn_up_feat 64 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180"
fi

if [ $model = "FSRCNN" ]; then
  # trained by: python main.py --n_GPUs 1 --scale 2 --patch_size 96 --batch_size 32 --data_test Set5 --lr 1e-3 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --model FSRCNN --reset --save fsrcnn_x2
  python test_runtime.py --n_threads 6 --cpu --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --n_colors 3 --model FSRCNN --save ../runtime_models/logs/fsrcnn_s64_x$3 --pre_train ../runtime_models/fsrcnn_s64_x$3.pt --test_only --reset --runtime --times $5 $6 #--no_count --save_result 
elif [ $model = "SRCNN" ]; then
  python test_runtime.py --n_threads 6 --cpu --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --n_colors 3 --model SRCNN --save ../runtime_models/logs/srcnn_s64_x$3 --pre_train ../runtime_models/srcnn_s64_x$3.pt --test_only --reset --runtime --times $5 $6 #--no_count --save_result 
elif [ $model = "v8" ]; then
  # ./scripts/train_srarn_v8.sh runtime 0 1 xt ab $3 48 ms skip v8old $5
  python test_runtime.py --n_threads 6 --cpu --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --n_colors 3 --res_connect skip $dim_configs --upsampling Nearest --acb_norm v8old --model SRARNV8 --save ../runtime_models/v8$2_x$3 --pre_train ../runtime_models/logs/v8$2_x$3.pt --test_only --load_inf --reset --runtime --times $5 $6 #--no_count --save_result 
elif [ $model = "v5" ]; then
  python test_runtime.py --n_threads 6 --cpu --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --n_colors 3 --res_connect 1acb3 $dim_configs --model SRARNV5 --save ../runtime_models/logs/v5$2_x$3 --pre_train ../runtime_models/v5$2_x$3.pt --test_only --load_inf --reset --runtime --times $5 $6 #--no_count --save_result 
elif [ $model = "SwinIR" ]; then
  python test_runtime.py --n_threads 6 --cpu --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --n_colors 3 --res_connect 1acb3 $dim_configs --model SwinIR --save ../runtime_models/logs/SwinIR-$2_x$3 --test_only --reset --runtime --times $5 $6 #--no_count --save_result 
elif [ $model = "EDSR-baseline" ]; then
  python test_runtime.py --n_threads 6 --cpu --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --n_colors 3 --model EDSR --save ../runtime_models/logs/EDSR-baseline_x$3 --pre_train ../models/EDSR-baseline_x$3.pt --test_only --reset --runtime --times $5 $6 #--no_count --save_result 
elif [ $model = "EDSR" ]; then
  python test_runtime.py --n_threads 6 --cpu --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --n_colors 3 --model EDSR --n_resblocks 32 --n_feats 256 --res_scale 0.1 --save ../runtime_models/logs/EDSR_x$3  --pre_train ../models/EDSR_x$3.pt --test_only --reset --runtime --times $5 $6 #--no_count --save_result 
else
  # echo "The model $1 is not supported!"
  python test_runtime.py --n_threads 6 --cpu --scale $3 --patch_size $patch --batch_size 32 --data_test 720P --n_colors 3 --model $model --save ../runtime_models/logs/${model}_x$3 --pre_train ../runtime_models/${model}_x$3.pt --test_only --reset --runtime --times $5 $6 #--no_count --save_result 
fi
