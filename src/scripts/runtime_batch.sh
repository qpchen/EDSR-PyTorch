#!/bin/bash

# models=("v5")
models=("SwinIR")
models2=("EDSR" "EDSR-baseline")
size_48=("b")
size_64=("s")
# size_64=("s" "t" "xt")
size=("no")
scales=(2 3 4)


for model in ${models2[@]}; do 
  for s in ${size[@]}; do 
    for scale in ${scales[@]}; do 
      echo "./scripts/test_runtime.sh $model $s $scale 48 50 \"--save_result --no_count\""
      ./scripts/test_runtime.sh $model $s $scale 48 50 "--save_result --no_count"
    done
  done
done

for model in ${models[@]}; do 
  for s in ${size_48[@]}; do 
    for scale in ${scales[@]}; do 
      echo "./scripts/test_runtime.sh $model $s $scale 48 50 \"--save_result --no_count\""
      ./scripts/test_runtime.sh $model $s $scale 48 50 "--save_result --no_count"
    done
  done
  for s in ${size_64[@]}; do 
    for scale in ${scales[@]}; do 
      echo "./scripts/test_runtime.sh $model $s $scale 64 50 \"--save_result --no_count\""
      ./scripts/test_runtime.sh $model $s $scale 64 50 "--save_result --no_count"
    done
  done
done
