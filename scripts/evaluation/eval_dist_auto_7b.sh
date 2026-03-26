#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_RT_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTHONPATH="./:$PYTHONPATH"

dataset=$1
split=${2:-"test"}

model_gnd_path="model_zoo/UFMA-7B"
model_checker_path="model_zoo/UFMA-7B"
model_pla_path="model_zoo/UFMA-7B"

pred_path="outputs_7b/${dataset}_${split}"

echo -e "\e[1;36mEvaluating:\e[0m $dataset ($split)"

python ufma/eval/infer_auto.py \
    --dataset $dataset \
    --split $split \
    --pred_path $pred_path \
    --model_gnd_path $model_gnd_path \
    --model_checker_path $model_checker_path \
    --model_pla_path $model_pla_path \
    --device all

python ufma/eval/eval_auto.py $pred_path --dataset $dataset
