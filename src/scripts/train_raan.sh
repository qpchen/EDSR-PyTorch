#!/bin/bash

################################################################################
######################      SRARN V9_D1acb3 noACBnorm befln nolr 2e-4       ######################
################################################################################
# ./scripts/train_srarn_v9.sh [mode] [cuda_device] [accummulation_step] [model_size] [interpolation] [sr_scale] [lr_patch_size] [LR_scheduler_class] [init LR] [stage Res] [acb_norm] [upsampling]
# run example for v9test_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0 1 test b 2 48 ms skip 1acb3 batch befln nolr 2e-4
# ########### training commands ########### _t: 38.069 327, 38.090 476, 38.108 711
# training: ./scripts/train_raan.sh train 0,1 1 s bc 2 48 ms 4e-4 noStageRes batch NN ACB 21

# training: ./scripts/train_raan.sh resume 1 1 t bc 2 48 ms 4e-4 noStageRes batch NN ACB 21
# training: ./scripts/train_raan.sh train 0 1 t bc 3 48 ms 4e-4 noStageRes batch NN ACB 21

# training t640: ./scripts/train_raan.sh resume 0 1 xt bc 2 48 ms 4e-4 noStageRes batch NN ACB 21
# training t640: ./scripts/train_raan.sh train 1 1 xt bc 3 48 ms 4e-4 noStageRes batch NN ACB 21

# ##### test add bilinear ######## bad _t: 38.069 327, 38.095 711
# giveup: ./scripts/train_raan.sh train 1 1 t bl 2 48 ms 4e-4 noStageRes batch NN ACB 21

# ##### test add nearest ######## best _t: 38.092 327, 38.105 476
# training: ./scripts/train_raan.sh train 0,1 1 s nr 2 48 ms 8e-4 noStageRes batch NN ACB 21
# waiting: ./scripts/train_raan.sh train 0,1 1 s nr 3 48 ms 8e-4 noStageRes batch NN ACB 21
# waiting: ./scripts/train_raan.sh train 0,1 1 s nr 4 48 ms 8e-4 noStageRes batch NN ACB 21

# training: ./scripts/train_raan.sh train 0,1 1 xs nr 2 48 ms 8e-4 noStageRes batch NN ACB 21

# training: ./scripts/train_raan.sh resume 0 1 t nr 2 48 ms 4e-4 noStageRes batch NN ACB 21
# training: ./scripts/train_raan.sh resume 1 1 t nr 3 48 ms 4e-4 noStageRes batch NN ACB 21
# training: ./scripts/train_raan.sh train 0 1 t nr 4 48 ms 4e-4 noStageRes batch NN ACB 21

# training: ./scripts/train_raan.sh train 1 1 xt nr 2 48 ms 4e-4 noStageRes batch NN ACB 21
# training: ./scripts/train_raan.sh train 0 1 xt nr 3 48 ms 4e-4 noStageRes batch NN ACB 21
# training: ./scripts/train_raan.sh train 1 1 xt nr 4 48 ms 4e-4 noStageRes batch NN ACB 21

  # ####ablation staty
  # 1 waiting: ./scripts/train_raan.sh train 0 1 xt nr 2 48 ms 4e-4 noStageRes batch NN noACB 21
  # 1 waiting: ./scripts/train_raan.sh train 1 1 xt nr 2 48 ms 4e-4 noStageRes inst NN ACB 21
  # 1 waiting: ./scripts/train_raan.sh train 2 1 xt nr 2 48 ms 4e-4 noStageRes no NN ACB 21
  # 2 waiting: ./scripts/train_raan.sh train 3 1 xt nr 2 48 ms 4e-4 noStageRes batch NN ACB 7
  # 2 waiting: ./scripts/train_raan.sh train 0 1 xt nr 2 48 ms 4e-4 noStageRes batch NN ACB 14
  # 2 waiting: ./scripts/train_raan.sh train 1 1 xt nr 2 48 ms 4e-4 noStageRes batch NN ACB 28
  # 3 waiting: ./scripts/train_raan.sh train 2 1 xt nr 2 48 ms 4e-4 useStageRes batch NN ACB 21
  # 4 waiting: ./scripts/train_raan.sh train 3 1 xt_mlp2 nr 2 48 ms 4e-4 noStageRes batch NN ACB 21
  # 6 waiting: ./scripts/train_raan.sh train 0 1 xt nr 2 48 ms 4e-4 noStageRes batch PSnA ACB 21
  # 6 waiting: ./scripts/train_raan.sh train 1 1 xt nr 2 48 ms 4e-4 noStageRes batch NNnPA ACB 21

# ##### test add pixel shuffle ######## _t: 38.089 327
# training: ./scripts/train_raan.sh train 3 1 t ps 2 48 ms 4e-4 noStageRes batch NN ACB 21
# waiting: ./scripts/train_raan.sh train 0 1 xt ps 2 48 ms 4e-4 noStageRes batch NN ACB 21

# ##### test small mlp_ratios ########
# waiting: ./scripts/train_raan.sh train 3 1 t_mlp2 bl 2 48 ms 4e-4 noStageRes batch NN ACB 21
# waiting: ./scripts/train_raan.sh train 0 1 xt_mlp2 bl 2 48 ms 4e-4 noStageRes batch NN ACB 21


# #####################################
# accept input
# first is run mode, 
mode=$1
# second is devices of gpu to use
device=$2
n_device=`expr ${#device} / 2 + 1`
# third is accumulation_step number
accum=$3
# forth is model size
size=$4
# ############## model_large #############
if [ $size = "l" ]; then
  options="--epochs 1000 --decay 500-800-900-950 --srarn_up_feat 180 --depths 8+8+8+8+8+8+8+8+8+8 --dims 180+180+180+180+180+180+180+180+180+180 --mlp_ratios 4+4+4+4+4+4+4+4+4+4 --batch_size 32"
# ############## model_base #############
elif [ $size = "b" ]; then
  options="--epochs 1000 --decay 500-800-900-950 --srarn_up_feat 180 --depths 6+6+6+6+6+6+6+6 --dims 180+180+180+180+180+180+180+180 --mlp_ratios 4+4+4+4+4+4+4+4 --batch_size 32"
# ############## model_small #############
elif [ $size = "s" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --srarn_up_feat 60 --depths 6+6+6+6+6 --dims 60+60+60+60+60 --mlp_ratios 4+4+4+4+4 --batch_size 32"
elif [ $size = "xs" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --srarn_up_feat 60 --depths 6+6+6+6 --dims 60+60+60+60 --mlp_ratios 4+4+4+4 --batch_size 32"
# ############## model_tiny #############
elif [ $size = "t" ]; then
  options="--epochs 2000 --decay 1000-1600-1800-1900 --srarn_up_feat 42 --depths 6+6+6 --dims 42+42+42 --mlp_ratios 4+4+4 --batch_size 32"
elif [ $size = "t_mlp2" ]; then
  options="--epochs 2000 --decay 1000-1600-1800-1900 --srarn_up_feat 42 --depths 6+6+6 --dims 42+42+42 --mlp_ratios 2+2+2 --batch_size 32"
# ############## model_xt extremely tiny #############
elif [ $size = "xt" ]; then
  options="--epochs 3000 --decay 1500-2400-2700-2850 --srarn_up_feat 24 --depths 6+6 --dims 24+24 --mlp_ratios 4+4 --batch_size 32"
elif [ $size = "xt_mlp2" ]; then
  options="--epochs 3000 --decay 1500-2400-2700-2850 --srarn_up_feat 24 --depths 6+6 --dims 24+24 --mlp_ratios 2+2 --batch_size 32"
# ############## test_model #############
elif [ $size = "test" ]; then  # test with lower costs
  options="--epochs 3000 --decay 1500-2400-2700-2850 --srarn_up_feat 6 --depths 2+4 --dims 6+12 --mlp_ratios 4+4 --batch_size 4"
else
  echo "no this size $size !"
  exit
fi
# if the output add interpolation of input
interpolation=$5
if [ $interpolation = "bc" ]; then
  interpolation_print=""
  interpolation=""
elif [ $interpolation = "bl" ]; then
  interpolation_print="_AddBL"
  interpolation="--interpolation Bilinear"
elif [ $interpolation = "nr" ]; then
  interpolation_print="_AddNr"
  interpolation="--interpolation Nearest"
elif [ $interpolation = "sk" ]; then
  interpolation_print="_AddSk"
  interpolation="--interpolation Skip"
elif [ $interpolation = "ps" ]; then
  interpolation_print="_AddPS"
  interpolation="--interpolation PixelShuffle"
else
  echo "no valid $interpolation ! Please input (bc | bl | nr | ps | sk)."
fi
# fifth is sr scale
scale=$6
# sixth is the LQ image patch size
patch=$7
patch_hr=`expr $patch \* $scale`
patch_print="_p$patch"
# lr_class choice, default is MultiStepLR. test whether CosineWarmRestart can be better
lr=$8
if [ $lr = "cosre" ]; then  # for CosineWarmRestart
  lr_class="CosineWarmRestart"
  lr_print="_CWRe"
elif [ $lr = "cos" ]; then  # for CosineWarm
  lr_class="CosineWarm"
  lr_print="_CW"
else  # $lr = "ms"
  lr_class="MultiStepLR"
  lr_print="_MS"
fi
initlr=$9
if [ $initlr = "2e-4" ]; then
  initlr_print=""
else
  initlr_print="_$initlr"
fi
# stage level residual connect
stageres=${10}
if [ $stageres = "useStageRes" ]; then
  stageres_opt="--stage_res"
  stageres_print="_StgRes"
elif [ $stageres = "noStageRes" ]; then  # better on test! better on other?
  stageres_opt=""
  stageres_print="_noStgRes"
else
  echo "no valid $stageres ! Please input (useStageRes | noStageRes)."
  exit
fi
# acb norm choices, can be "batch", "inst", "no", "v8old"
acb=${11}
acb_print="_ACB$acb"
# upsampling optionsNearestNoPA
upsam=${12}
if [ $upsam = "NN" ]; then  # best? use Nearest-Neibor
  upsam_print="_UpNN"
  upsam_opt="Nearest"
elif [ $upsam = "PSnA" ]; then  # nA better? use PixelShuffle with no activate layer, same as SwinIR
  upsam_print="_UpPSnA"
  upsam_opt="PixelShuffle --no_act_ps"
elif [ $upsam = "PS" ]; then  # worst? use PixelShuffle with activate layer
  upsam_print="_UpPS"
  upsam_opt="PixelShuffle"
elif [ $upsam = "NNnPA" ]; then  # worse? use Nearest-Neibor without pixel attention
  upsam_print="_UpNNnPA"
  upsam_opt="NearestNoPA"
else
  echo "no valid $upsam ! Please input (NN | PS | PSnA | NNnPA)."
  exit
fi
# use ACB or not
use_acb=${13}
if [ $use_acb = "ACB" ]; then  # best? use Nearest-Neibor
  use_acb_print=""
  use_acb_opt="--use_acb"
elif [ $use_acb = "noACB" ]; then 
  use_acb_print="_noACB"
  use_acb_opt=""
else
  echo "no valid $use_acb ! Please input (ACB | noACB)."
  exit
fi
# set large kernel (LKA) size
LKAk=${14}
if [ $LKAk = "21" ]; then  # default
  LKAk_print=""
  LKAk_opt=""
elif [ $LKAk = "14" ]; then 
  LKAk_print="_LK14"
  LKAk_opt="--DWDkSize 7 --DWDdil 2"
elif [ $LKAk = "7" ]; then 
  LKAk_print="_LK7"
  LKAk_opt="--DWDkSize 7 --DWDdil 1"
elif [ $LKAk = "28" ]; then 
  LKAk_print="_LK28"
  LKAk_opt="--DWDkSize 7 --DWDdil 4"
else
  echo "no valid $LKAk ! Please input (7 | 14 | 21 | 28)."
  exit
fi


# #####################################
# prepare program options parameters
# v9 must use layernorm
run_command="python main.py --n_GPUs $n_device --accumulation_step $accum --scale $scale --patch_size $patch_hr $options $interpolation --acb_norm $acb $stageres_opt --upsampling $upsam_opt --loss 1*SmoothL1 --lr $initlr --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --lr_class $lr_class $use_acb_opt $LKAk_opt --model RAAN"
father_dir="../raan${upsam_print}${use_acb_print}${acb_print}${stageres_print}${interpolation_print}${lr_print}${initlr_print}"
file_name="v1${size}${patch_print}${LKAk_print}_x${scale}"
save_dir="${father_dir}/${file_name}"
log_file="${father_dir}/logs/${file_name}.log"

if [ ! -d "${father_dir}" ]; then
  mkdir "${father_dir}"
fi
if [ ! -d "${father_dir}/logs" ]; then
  mkdir "${father_dir}/logs"
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

