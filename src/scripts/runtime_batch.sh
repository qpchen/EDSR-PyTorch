#!/bin/bash

# ./scripts/runtime_batch.sh "RACN" 48 "no" "2 3 4" "--save_result --no_count --n_GPUs 1"
# ./scripts/runtime_batch.sh "RACN" 48 "no" "2 3 4" "--n_GPUs 1"
# ./scripts/runtime_batch.sh "RACN" 48 "no" "2 3 4" "--save_result --no_count --cpu"
# ./scripts/runtime_batch.sh "LBNet LBNet-T" 48 "no" "2 3 4" "--save_result --no_count --cpu --times 3"
# ./scripts/runtime_batch.sh "v5" 64 "fblt fbt fbxt" "2 3 4" "--save_result --no_count --cpu --times 50"
# ./scripts/runtime_batch.sh "FSRCNN SRCNN" 64 "no" "2 3 4" "--save_result --no_count --n_GPUs 1 --times 50"
# ./scripts/runtime_batch.sh "IMDN LAPAR_A LAPAR_B LAPAR_C CARN LatticeNet LBNet LBNet-T ESRT IDN LapSRN DRRN EDSR-baseline RCAN" 48 "no" "2 3 4" "--save_result --no_count --n_GPUs 1 --times 50"
# ./scripts/runtime_batch.sh "LBNet LBNet-T ESRT IDN LapSRN DRRN" 48 "no" "2 3 4" "--save_result --no_count --n_GPUs 1 --times 50"
# ./scripts/runtime_batch.sh "v9" 64 "fblt fbxt" "2 3 4" "--save_result --no_count --n_GPUs 1 --times 50"
# ./scripts/runtime_batch.sh "v9" 48 "fs" "2 3 4" "--save_result --no_count --n_GPUs 1 --times 50"
# ./scripts/runtime_batch.sh "ALAN" 48 "v1xt v1t v1xs v1s" "2 3 4" "--save_result --no_count --n_GPUs 1 --times 50"
# ./scripts/runtime_batch.sh "ALAN" 48 "v1xt v1t v1xs" "2 3 4" "--save_result --no_count --n_GPUs 1 --times 50"
# ./scripts/runtime_batch.sh "SwinIR" 64 "b s" "2 3 4" "--save_result --no_count --n_GPUs 1 --times 50"
# ./scripts/runtime_batch.sh "ShuffleMixer" 48 "base tiny" "2 3 4" "--save_result --no_count --n_GPUs 1 --times 50"
# ./scripts/runtime_batch.sh "DCANV1" 48 "t2" "2" "--save_result --no_count --n_GPUs 1 --times 50"
# ./scripts/runtime_batch.sh "DCANV2" 48 "xt2 t2 b26" "2 3 4" "--save_result --no_count --n_GPUs 1 --times 50"
# ./scripts/runtime_batch.sh "DCANV2" 48 "xt2 t2 b26" "2 3 4" "--save_result --no_count --n_GPUs 1 --times 1 --data_test Demo --dir_demo ../../dataset/historical/LR"

models=($1)
patch=$2
sizes=($3)
# scales=(2 3 4)
scales=($4)

for model in ${models[@]}; do 
#  echo "$model"
  for s in ${sizes[@]}; do 
  # echo "$s"
    for scale in ${scales[@]}; do 
      # echo "$scale"
      # if [ $patch = "48" ]; then
        echo "./scripts/test_runtime.sh $model $s $scale $patch 50 \"$5\""
        ./scripts/test_runtime.sh $model $s $scale $patch 50 "$5"
      # else
      #   echo "./scripts/test_runtime.sh $model $s $scale 64 50 \"$5\""
      #   ./scripts/test_runtime.sh $model $s $scale 64 50 "$5"
      # fi
    done
  done
done

# models1=("SwinIR")
# models2=("EDSR" "EDSR-baseline")
# models3=("RCAN")
# models4=("v5")
# size_48=("b")
# size_64=("s")
# # size_64=("s" "t" "xt")
# size=("no")

# for model in ${models3[@]}; do 
#   for s in ${size[@]}; do 
#     for scale in ${scales[@]}; do 
#       echo "./scripts/test_runtime.sh $model $s $scale 48 50 \"--save_result --no_count --cpu\""
#       ./scripts/test_runtime.sh $model $s $scale 48 50 "--save_result --no_count --cpu"
#     done
#   done
# done

# for model in ${models2[@]}; do 
#   for s in ${size[@]}; do 
#     for scale in ${scales[@]}; do 
#       echo "./scripts/test_runtime.sh $model $s $scale 48 50 \"--save_result --no_count --cpu\""
#       ./scripts/test_runtime.sh $model $s $scale 48 50 "--save_result --no_count --cpu"
#     done
#   done
# done

# for model in ${models[@]}; do 
#   for s in ${size_48[@]}; do 
#     for scale in ${scales[@]}; do 
#       # echo "./scripts/test_runtime.sh $model $s $scale 48 50 \"--save_result --no_count --cpu\""
#       # ./scripts/test_runtime.sh $model $s $scale 48 50 "--save_result --no_count --cpu"
#       echo "./scripts/test_runtime.sh $model $s $scale 48 50 \"--n_GPUs 1\""
#       ./scripts/test_runtime.sh $model $s $scale 48 1 "--n_GPUs 1"
#     done
#   done
#   # for s in ${size_64[@]}; do 
#   #   for scale in ${scales[@]}; do 
#   #     echo "./scripts/test_runtime.sh $model $s $scale 64 50 \"--save_result --no_count --cpu\""
#   #     ./scripts/test_runtime.sh $model $s $scale 64 50 "--save_result --no_count --cpu"
#   #   done
#   # done
# done

