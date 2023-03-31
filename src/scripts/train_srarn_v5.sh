#!/bin/bash

################################################################################
######################      SRARN V5       ######################
################################################################################
# ./scripts/train_srarn_v5.sh [mode] [cuda_device] [accummulation_step] [model] [interpolation] [sr_scale] [lr_patch_size] [LR_scheduler_class] [block_conv] [deep_conv] [acb_norm] [LN]
# run example for v5test_x2: ./scripts/train_srarn_v5.sh train 0 1 test s 2 48 ms 1acb3
# ########### training commands ###########

# run example for v5bn_x3: ./scripts/train_srarn_v5.sh train 0,1 1 bn b 3 48 ms 1acb3
# run example for v5bn_x4: ./scripts/train_srarn_v5.sh train 2,3 1 bn b 4 48 ms 1acb3

# run example for v5bn_x2: ./scripts/train_srarn_v5.sh train 0,1,2,3 1 bn b 2 48 ms 1acb3
# run example for v5b_Nrst_x2: ./scripts/train_srarn_v5.sh train 0,1,2 1 b n 2 48 ms 1acb3
# run example for v5b_PxSh_x2: ./scripts/train_srarn_v5.sh train 0 4 b p 2 48 ms 1acb3
# run example for v5ba_x2: ./scripts/train_srarn_v5.sh train 0,1 1 ba b 2 48 ms 1acb3


# run example for v5s_Nrst_x2: ./scripts/train_srarn_v5.sh train 0 1 s n 2 64 ms 1acb3
# run example for v5s_PxSh_x2: ./scripts/train_srarn_v5.sh train 3 1 s p 2 64 ms 1acb3

# run example for v5s_x2: ./scripts/train_srarn_v5.sh train 0 1 s b 2 64 ms 1acb3
# run example for v5s_x3: ./scripts/train_srarn_v5.sh train 0,1 1 s b 3 64 ms 1acb3
# run example for v5s_x4: ./scripts/train_srarn_v5.sh train 0,1 1 s b 4 64 ms 1acb3
# run example for v5t_x3: ./scripts/train_srarn_v5.sh train 0 1 t b 3 64 ms 1acb3
# run example for v5t_x4: ./scripts/train_srarn_v5.sh train 0 1 t b 4 64 ms 1acb3

# run example for v5t_x2: ./scripts/train_srarn_v5.sh train 0 1 t b 2 64 ms 1acb3
# run example for v5xt_x2: ./scripts/train_srarn_v5.sh train 0 1 xt b 2 64 ms 1acb3
# run example for v5xt_x3: ./scripts/train_srarn_v5.sh train 1 1 xt b 3 64 ms 1acb3
# run example for v5xt_x4: ./scripts/train_srarn_v5.sh train 1 1 xt b 4 64 ms 1acb3


# run example for v5xt_x2: ./scripts/train_srarn_v5.sh train 1 1 xt b 2 64 ms 3acb3

# run example for v5lt_x2: ./scripts/train_srarn_v5.sh train 1 1 lt b 2 64 ms 1acb3
# run example for v5lt_x3: ./scripts/train_srarn_v5.sh train 1 1 lt b 3 64 ms 1acb3
# run example for v5lt_x4: ./scripts/train_srarn_v5.sh train 1 1 lt b 4 64 ms 1acb3

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
elif [ $size = "fblt" ]; then
  options="--epochs 2000 --decay 1000-1600-1800-1900 --upsampling Nearest --srarn_up_feat 42 --depths 6+6+6 --dims 42+42+42+42 --batch_size 32"
# ############## model_t #############
elif [ $size = "t" ]; then
  options="--epochs 2000 --decay 1000-1600-1800-1900 --upsampling Nearest --srarn_up_feat 30 --depths 3+3+3+3 --dims 30+30+30+30 --batch_size 32"
# ############## fixed block model_t #############
elif [ $size = "fbt" ]; then
  options="--epochs 2000 --decay 1000-1600-1800-1900 --upsampling Nearest --srarn_up_feat 30 --depths 6+6 --dims 30+30+30+30 --batch_size 32"
# ############## model_xt #############
elif [ $size = "xt" ]; then
  options="--epochs 3000 --decay 1500-2400-2700-2850 --upsampling Nearest --srarn_up_feat 24 --depths 2+2+2+2 --dims 24+24+24+24 --batch_size 32"
# ############## fixed block model_xt #############
elif [ $size = "fbxt" ]; then
  options="--epochs 3000 --decay 1500-2400-2700-2850 --upsampling Nearest --srarn_up_feat 24 --depths 6+6 --dims 24+24+24+24 --batch_size 32"
# ############## test_model #############
elif [ $size = "test" ]; then  # test with lower costs
  options="--epochs 3000 --decay 1500-2400-2700-2850 --upsampling Nearest --srarn_up_feat 6 --depths 2+4 --dims 6+12 --batch_size 4"
else
  echo "no this size $size !"
  exit
fi
# if the output add bicubic interpolation of input
interpolation=$5
if [ $interpolation = "b" ]; then
  interpolation_print=""
  interpolation=""
elif [ $interpolation = "n" ]; then
  interpolation_print="_Nrst"
  interpolation="--interpolation Nearest"
elif [ $interpolation = "s" ]; then
  interpolation_print="_Skip"
  interpolation="--interpolation Skip"
elif [ $interpolation = "p" ]; then
  interpolation_print="_PxSh"
  interpolation="--interpolation PixelShuffle"
else
  echo "no valid $interpolation ! Please input (b | n | s)."
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
# res_connect choice, other version default is 1acb3(same as version 5). 
res=$9
if [ $res = "1acb3" ]; then
  res_print=""
else
  res_print="_$res"
fi



# #####################################
# prepare program options parameters
# v5 must use layernorm
run_command="python main.py --n_GPUs $n_device --accumulation_step $accum --scale $scale --patch_size $patch_hr $options $interpolation --data_range 1-800 --res_connect $res --loss 1*L1 --lr 2e-4 --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --lr_class $lr_class --model SRARNV5"

save_dir="../srarn_v5/v5${size}${patch_print}${interpolation_print}${res_print}${lr_print}_x${scale}"
log_file="../srarn_v5/logs/v5${size}${patch_print}${interpolation_print}${res_print}${lr_print}_x${scale}.log"

if [ ! -d "../srarn_v5" ]; then
  mkdir "../srarn_v5"
fi
if [ ! -d "../srarn_v5/logs" ]; then
  mkdir "../srarn_v5/logs"
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

