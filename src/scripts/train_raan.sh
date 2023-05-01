#!/bin/bash

################################################################################
######################      SRARN V9_D1acb3 noACBnorm befln nolr 2e-4       ######################
################################################################################
# ./scripts/train_srarn_v9.sh [mode] [cuda_device] [accummulation_step] [model_size] [interpolation] [sr_scale] [lr_patch_size] [LR_scheduler_class] [init LR] [stage Res] [acb_norm] [upsampling]
# run example for v9test_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0 1 test b 2 48 ms skip 1acb3 batch befln nolr 2e-4
# ########### training commands ###########
# ./scripts/train_raan.sh train 0 1 s b 2 48 ms 4e-4 noStageRes batch NN # training


# ./scripts/train_raan.sh train 0 1 xt b 2 48 ms 4e-4 noStageRes batch NN  # training t640

# ################### test: 1acb3 head; BN; before LN;  ######################
# for batch_befLN/v9l2_1acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0 1 l2 b 2 48 ms 1acb3 1acb3 batch befln nolr 2e-4  # training shen a100

# for batch_befLN/v9l3_1acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0,1,2,3 1 l3 b 2 48 ms 1acb3 1acb3 batch befln nolr 2e-4  # training shen  # larger batch for multi-GPU
# for batch_befLN/v9l3_1acb3_D1acb3_x3: ./scripts/train_srarn_v9.sh train 0,1,2,3 1 l3 b 3 48 ms 1acb3 1acb3 batch befln nolr 2e-4  # training shen  # larger batch for multi-GPU
# for batch_befLN/v9l3_1acb3_D1acb3_x4: ./scripts/train_srarn_v9.sh train 0,1,2,3 2 l3 b 4 48 ms 1acb3 1acb3 batch befln nolr 2e-4  # training shen  # larger batch for multi-GPU
# for batch_befLN/v9l3_1acb3_D1acb3_x4: ./scripts/train_srarn_v9.sh train 0 2 l3 b 4 48 ms 1acb3 1acb3 batch befln nolr 4e-4  # training shen a100  # larger lr


# for batch_befLN/v9fbxt_s64_1acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0 1 fbxt b 2 64 ms 1acb3 1acb3 batch befln nolr 2e-4  # training shen
# for batch_befLN/v9fbxt_s64_1acb3_D1acb3_x3: ./scripts/train_srarn_v9.sh train 1 1 fbxt b 3 64 ms 1acb3 1acb3 batch befln nolr 2e-4  # training shen
# for batch_befLN/v9fbxt_s64_1acb3_D1acb3_x4: ./scripts/train_srarn_v9.sh train 0 1 fbxt b 4 64 ms 1acb3 1acb3 batch befln nolr 2e-4  # training shen
# for batch_befLN/v9fbxt_s64_3acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0 1 fbxt b 2 64 ms 3acb3 1acb3 batch befln nolr 2e-4  # training t640  # for param less than 100K
# for batch_befLN/v9fbxt_s64_3acb3_D1acb3_x3: ./scripts/train_srarn_v9.sh train 1 1 fbxt b 3 64 ms 3acb3 1acb3 batch befln nolr 2e-4  # training t640
# for batch_befLN/v9fbxt_s64_3acb3_D1acb3_x4: ./scripts/train_srarn_v9.sh train 0 1 fbxt b 4 64 ms 3acb3 1acb3 batch befln nolr 2e-4  # training shen
# for batch_befLN/v9fblt_s64_1acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh train 1 1 fblt b 2 64 ms 1acb3 1acb3 batch befln nolr 2e-4  # training shen
# for batch_befLN/v9fblt_s64_1acb3_D1acb3_x3: ./scripts/train_srarn_v9.sh train 2 1 fblt b 3 64 ms 1acb3 1acb3 batch befln nolr 2e-4  # training shen
# for batch_befLN/v9fblt_s64_1acb3_D1acb3_x4: ./scripts/train_srarn_v9.sh train 3 1 fblt b 4 64 ms 1acb3 1acb3 batch befln nolr 2e-4  # training shen
# for batch_befLN/v9fs_1acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh train 1 1 fs b 2 48 ms 1acb3 1acb3 batch befln nolr 2e-4  # training shen
# for batch_befLN/v9fs_1acb3_D1acb3_x3: ./scripts/train_srarn_v9.sh train 1 1 fs b 3 48 ms 1acb3 1acb3 batch befln nolr 2e-4  # training shen
# for batch_befLN/v9fs_1acb3_D1acb3_x4: ./scripts/train_srarn_v9.sh train 0 1 fs b 4 48 ms 1acb3 1acb3 batch befln nolr 2e-4  # training shen



# for batch_befLN/v9fbxt_1acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh train 1 1 fbxt b 2 48 ms 1acb3 1acb3 batch befln nolr 2e-4  # pause shen  # bad for p48
# for batch_befLN/v9fblt_1acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0 1 fblt b 2 48 ms 1acb3 1acb3 batch befln nolr 2e-4  # pause shen  # bad for p48
# for batch/v9fbxt_s64_1acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh train 1 1 fbxt b 2 64 ms 1acb3 1acb3 batch ln nolr 2e-4  # done shen  # bad 37.899  p48:37.871
# for batch_noLN/v9fbxt_s64_1acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh train 1 1 fbxt b 2 64 ms 1acb3 1acb3 batch no nolr 2e-4  # done shen  # better 37.919 p48:37.903
# for batch/v9fblt_s64_1acb3_D1acb3_x2: ./scripts/train_srarn_v9.sh train 0 1 fblt b 2 64 ms 1acb3 1acb3 batch ln nolr 2e-4  # waiting shen

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
# ############## model_normal #############
elif [ $size = "n" ]; then  # model_b use PixelShuffle upsampling with no activate layer, same as SwinIR
  options="--epochs 1000 --decay 500-800-900-950 --srarn_up_feat 180 --depths 6+6+6+6+6+6+6+6 --dims 180+180+180+180+180+180+180+180 --mlp_ratios 4+4+4+4+4+4+4+4 --batch_size 32"
# ############## model_small #############
elif [ $size = "s" ]; then
  options="--epochs 1500 --decay 750-1200-1350-1425 --srarn_up_feat 60 --depths 6+6+6+6+6 --dims 60+60+60+60+60 --mlp_ratios 4+4+4+4+4 --batch_size 32"
# ############## model_tiny #############
elif [ $size = "t" ]; then
  options="--epochs 2000 --decay 1000-1600-1800-1900 --srarn_up_feat 42 --depths 6+6+6 --dims 42+42+42 --mlp_ratios 4+4+4 --batch_size 32"
# ############## model_xt extremely tiny #############
elif [ $size = "xt" ]; then
  options="--epochs 3000 --decay 1500-2400-2700-2850 --srarn_up_feat 24 --depths 6+6 --dims 24+24 --mlp_ratios 4+4 --batch_size 32"
# ############## test_model #############
elif [ $size = "test" ]; then  # test with lower costs
  options="--epochs 3000 --decay 1500-2400-2700-2850 --srarn_up_feat 6 --depths 2+4 --dims 6+12 --mlp_ratios 4+4 --batch_size 4"
else
  echo "no this size $size !"
  exit
fi
# if the output add interpolation of input
interpolation=$5
if [ $interpolation = "b" ]; then  # use bicubic interpolation
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
  exit
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
  lr_print="_CR"
elif [ $lr = "cos" ]; then  # for CosineWarm
  lr_class="CosineWarm"
  lr_print="_C"
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
  stageres_print="_useStgRes"
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
# upsampling options
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
else
  echo "no valid $upsam ! Please input (NN | PS | PSnA)."
  exit
fi


# #####################################
# prepare program options parameters
# v9 must use layernorm
run_command="python main.py --n_GPUs $n_device --accumulation_step $accum --scale $scale --patch_size $patch_hr $options $interpolation --acb_norm $acb $stageres_opt --upsampling $upsam_opt --loss 1*SmoothL1 --lr $initlr --n_colors 3 --optimizer ADAM --skip_threshold 1e6 --lr_class $lr_class --use_acb --model RAAN"
father_dir="../raan${upsam_print}${acb_print}${stageres_print}${interpolation_print}${lr_print}${initlr_print}"
file_name="v1${size}${patch_print}_x${scale}"
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

