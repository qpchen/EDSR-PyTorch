#!/bin/bash

# ./scripts/runtime_batch.sh "RACN" "no" "2 3 4" "--save_result --no_count --n_GPUs 1"
# ./scripts/runtime_batch.sh "RACN" "no" "2 3 4" "--n_GPUs 1"
# ./scripts/runtime_batch.sh "RACN" "no" "2 3 4" "--save_result --no_count --cpu"
# ./scripts/runtime_batch.sh "LBNet LBNet-T" "no" "2 3 4" "--save_result --no_count --cpu --times 3"

models=($1)
sizes=($2)
# scales=(2 3 4)
scales=($3)

for model in ${models[@]}; do 
#  echo "$model"
  for s in ${sizes[@]}; do 
  # echo "$s"
    for scale in ${scales[@]}; do 
      # echo "$scale"
      if [ $s = "b" -o $s = "no" ]; then
        echo "./scripts/test_runtime.sh $model $s $scale 48 50 \"$4\""
        ./scripts/test_runtime.sh $model $s $scale 48 50 "$4"
      else
        echo "./scripts/test_runtime.sh $model $s $scale 64 50 \"$4\""
        ./scripts/test_runtime.sh $model $s $scale 64 50 "$4"
      fi
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

