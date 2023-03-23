#!/bin/bash

################################################################################
######################      SRARN V9_D1acb3 noACBnorm befln nolr       ######################
################################################################################
# ./scripts/train_srarn_v9.sh [mode] [cuda_device] [accummulation_step] [model] [use_bicubic] [sr_scale] [lr_patch_size] [LR_scheduler_class] [block_conv] [deep_conv] [acb_norm] [LN] [addLR]
# run example for v9test_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0 1 test ab 2 48 ms skip 1acb3 batch befln nolr
# ########### training commands ###########
# ###### on Starlight: ######
# run example for v9ba_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0,1 1 ba ab 2 48 ms skip 1acb3 batch befln nolr  # 38.243 v8  # PixelShuffleAct+BicubicAdd
# run example for v9b_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0,1 1 b ab 2 48 ms skip 1acb3 batch befln nolr  # 38.257 v8  # PixelShuffleNOAct+Bicubic

# run example for v9bn_D1acb3_x2: ./scripts/train_srarn_v9.sh train 2,3 1 bn ab 2 48 ms skip 1acb3 batch befln nolr  # 38.266 v8: Nearest+BicubicAdd 
# run example for v9bn_D1acb3_x3: ./scripts/train_srarn_v9.sh train 0,1 1 bn ab 3 48 ms skip 1acb3 batch befln nolr
# run example for v9bn_D1acb3_x4: ./scripts/train_srarn_v9.sh train 0,1 1 bn ab 4 48 ms skip 1acb3 batch befln nolr

# run example for v9s_s64_D1acb3_x2: ./scripts/train_srarn_v9.sh train 1 1 s ab 2 64 ms skip 1acb3 inst befln nolr  # comparing
# run example for v9s_s64_D1acb3_x2: ./scripts/train_srarn_v9.sh train 1 1 s ab 2 64 ms skip 1acb3 batch befln nolr
# run example for v9s_s64_D1acb3_x2: ./scripts/train_srarn_v9.sh train 1 1 s ab 2 64 ms skip 1acb3 batch befln addlr  # comparing

# run example for v9s_s64_D1acb3_x3: ./scripts/train_srarn_v9.sh train 1 1 s ab 3 64 ms skip 1acb3 batch befln nolr
# run example for v9s_s64_D1acb3_x4: ./scripts/train_srarn_v9.sh train 1 1 s ab 4 64 ms skip 1acb3 batch befln nolr

# run example for v9lt_s64_D3acb3_x2: ./scripts/train_srarn_v9.sh train 0 1 lt ab 2 64 ms skip 1acb3 batch befln nolr
# run example for v9lt_s64_D3acb3_x3: ./scripts/train_srarn_v9.sh train 1 1 lt ab 3 64 ms skip 1acb3 batch befln nolr
# run example for v9lt_s64_D3acb3_x4: ./scripts/train_srarn_v9.sh train 1 1 lt ab 4 64 ms skip 1acb3 batch befln nolr

# run example for v9t_s64_D1acb3_x4: ./scripts/train_srarn_v9.sh train 0 1 t ab 4 64 ms skip 1acb3 batch befln nolr
# run example for v9t_s64_D1acb3_x3: ./scripts/train_srarn_v9.sh train 0 1 t ab 3 64 ms skip 1acb3 batch befln nolr

# ###### on t640: ######
# run example for v9t_s64_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0 1 t ab 2 64 ms skip 1acb3 batch befln nolr
# run example for v9xt_s64_D3acb3_x2: ./scripts/train_srarn_v9.sh train 0 1 xt ab 2 64 ms skip 1acb3 batch befln nolr
# run example for v9xt_s64_D3acb3_x3: ./scripts/train_srarn_v9.sh train 1 1 xt ab 3 64 ms skip 1acb3 batch befln nolr
# run example for v9xt_s64_D3acb3_x4: ./scripts/train_srarn_v9.sh train 1 1 xt ab 4 64 ms skip 1acb3 batch befln nolr

# run example for v9xt_s64_D3acb3_x2: ./scripts/train_srarn_v9.sh train 0 1 xt ab 2 64 ms skip 1acb3 batch befln addlr  # comparing

# #####################################
# accept input
# input 2 params, first is run mode, 
mode=$1
# second is devices of gpu to use
device=$2
n_device=`expr ${#device} / 2 + 1`
# third is accumulation_step number
accum=$3
# forth is model size
size=$4
# ############## model_b #############
if [ $size = "b" ]; then  # model_b use PixelShuffle upsampling with no activate layer, same as SwinIR
  options="--epochs 1000 --decay 500-800-900-950 --upsampling PixelShuffle --no_act_ps --srarn_up_feat 64 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --batch_size 32"
elif [ $size = "ba" ]; then  # model_b use PixelShuffle upsampling with activate layer, same as version 5
  options="--epochs 1000 --decay 500-800-900-950 --upsampling PixelShuffle --srarn_up_feat 64 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --batch_size 32"
elif [ $size = "bn" ]; then  # model_b with nearest+conv upsampling
  options="--epochs 1000 --decay 500-800-900-950 --upsampling Nearest --srarn_up_feat 64 --depths 6+6+6+6+6+6 --dims 180+180+180+180+180+180 --batch_size 32"
# ############## model_s #############
elif [ $size = "s" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --upsampling Nearest --srarn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --batch_size 32"
# ############## model_lt larger tiny #############
elif [ $size = "lt" ]; then
  options="--epochs 2000 --decay 1000-1600-1800-1900 --upsampling Nearest --srarn_up_feat 42 --depths 4+4+4+4 --dims 42+42+42+42 --batch_size 32"
# ############## model_t #############
elif [ $size = "t" ]; then
  options="--epochs 2000 --decay 1000-1600-1800-1900 --upsampling Nearest --srarn_up_feat 30 --depths 3+3+3+3 --dims 30+30+30+30 --batch_size 32"
# ############## model_xt #############
elif [ $size = "xt" ]; then
  options="--epochs 3000 --decay 1500-2400-2700-2850 --upsampling Nearest --srarn_up_feat 24 --depths 2+2+2+2 --dims 24+24+24+24 --batch_size 32"
# ############## test_model #############
elif [ $size = "test" ]; then  # test with lower costs
  options="--epochs 3000 --decay 1500-2400-2700-2850 --upsampling Nearest --srarn_up_feat 6 --depths 2+4 --dims 6+12 --batch_size 4"
else
  echo "no this size $size !"
  exit
fi
# if the output add bicubic interpolation of input
use_bicubic=$5
if [ $use_bicubic = "ab" ]; then
  bicubic_print=""
  bicubic=""
elif [ $use_bicubic = "nb" ]; then
  bicubic_print="_nb"
  bicubic="--no_bicubic"
else
  echo "no valid $use_bicubic ! Please input (ab | nb)."
fi
# fifth is sr scale
scale=$6
# sixth is the LQ image patch size
patch=$7
patch_hr=`expr $patch \* $scale`
if [ $patch = 48 ]; then
  patch_print=""
else
  patch_print="_s$patch"
fi

# lr_class choice, default is MultiStepLR. test whether CosineWarmRestart can be better
# if [ $# == 8 ]; then
  lr=$8
  if [ $lr = "cosre" ]; then  # for CosineWarmRestart
    lr_class="CosineWarmRestart"
    lr_print="_CR"
  elif [ $lr = "cos" ]; then  # for CosineWarm
    lr_class="CosineWarm"
    lr_print="_C"
  else  # $lr = "ms"
    lr_class="MultiStepLR"
    lr_print=""
  fi
# else
#   lr_class="MultiStepLR"
#   lr_print=""
# fi
# res_connect choice, other version default is 1acb3(same as version 5). 
# But for v9 default 'skip'. other choices are 1conv1 3acb3
res=$9
if [ $res = "skip" ]; then
  res_print=""
else
  res_print="_$res"
fi
deep=${10}
# the last conv at end of deep feature module, before skip connect
if [ $deep = "skip" ]; then
  deep_print=""
else
  deep_print="_D$deep"
fi
# acb norm choices, can be "batch", "inst", "no", "v8old"
acb=${11}
acb_print="_$acb"
# backbone norm choices
norm=${12}
if [ $norm = "ln" ]; then
  norm_opt="--norm_at after"
  norm_print=""
elif [ $norm = "no" ]; then  # "no": means do not use norm
  norm_opt="--no_layernorm"
  norm_print="_noLN"
elif [ $norm = "befln" ]; then
  norm_opt="--norm_at before"
  norm_print="_befLN"
fi
# add lr to upsampling, can be "addlr" or "nolr"
addlr=${13}
if [ $addlr = "addlr" ]; then
  addlr_opt="--add_lr"
  addlr_print="_addlr"
elif [ $addlr = "nolr" ]; then
  addlr_opt=""
  addlr_print=""
fi


# #####################################
# prepare program options parameters
# v9 must use layernorm
run_command="python main.py --n_GPUs $n_device --accumulation_step $accum --scale $scale --patch_size $patch_hr $options $bicubic --res_connect $res --deep_conv $deep --acb_norm $acb --loss 1*L1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --lr_class $lr_class $norm_opt $addlr_opt --model SRARNV9"
# run_command="python main.py --n_GPUs $n_device --accumulation_step $accum --scale $scale --patch_size $patch_hr $options $bicubic --res_connect $res --deep_conv $deep --loss 1*L1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --lr_class CosineWarmRestart --model SRARNV9"
save_dir="../srarn_v9${acb_print}${norm_print}/v9${size}${patch_print}${addlr_print}${bicubic_print}${res_print}${deep_print}${lr_print}_x${scale}"
log_file="../srarn_v9${acb_print}${norm_print}/logs/v9${size}${patch_print}${addlr_print}${bicubic_print}${res_print}${deep_print}${lr_print}_x${scale}.log"

if [ ! -d "../srarn_v9${acb_print}${norm_print}" ]; then
  mkdir "../srarn_v9${acb_print}${norm_print}"
fi
if [ ! -d "../srarn_v9${acb_print}${norm_print}/logs" ]; then
  mkdir "../srarn_v9${acb_print}${norm_print}/logs"
fi


# #####################################
# run train/eval program
export CUDA_VISIBLE_DEVICES=$device
echo "CUDA GPUs use: No.'$CUDA_VISIBLE_DEVICES' devices."

if [ $mode = "train" ]
then
  if [ -f "$save_dir/model/model_latest.pt" ]; then
    echo "$save_dir seems storing some model files trained before, please change the save dir!"
  else
    echo "start training from the beginning:"
    echo "nohup $run_command --save $save_dir --reset > $log_file 2>&1 &"
    nohup $run_command --save $save_dir --reset > $log_file 2>&1 &
  fi
elif [ $mode = "resume" ]
then
  echo "resume training:"
  echo "nohup $run_command --load $save_dir --resume -1 > $log_file 2>&1 &"
  nohup $run_command --load $save_dir --resume -1 >> $log_file 2>&1 &
elif [ $mode = "switch" ]
then
  echo "switch acb from training to inference mode:"
  echo "$run_command --save ${save_dir}_test --pre_train $save_dir/model/model_best.pt --test_only --inf_switch"
  $run_command --save ${save_dir}_test --pre_train $save_dir/model/model_best.pt --test_only --inf_switch
elif [ $mode = "eval" ]
then
  echo "load inference version of acb to eval:"
  echo "$run_command --data_test Set5+Set14+B100+Urban100+Manga109 --save ${save_dir}_test --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf"
  $run_command --data_test Set5+Set14+B100+Urban100+Manga109 --save ${save_dir}_test --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf
elif [ $mode = "runtime" ]
then
  # echo "load inference version of acb to test the runtime:"
  # echo "$run_command --data_test 720P --runtime --no_count --save ${save_dir}_test --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf"
  $run_command --data_test 720P --runtime --no_count --save ${save_dir}_test --pre_train ${save_dir}_test/model/inf_model.pt --test_only --save_result --load_inf --times ${11}
else
  echo "invalid value, it only accpet train, resume, switch, eval!"
fi
