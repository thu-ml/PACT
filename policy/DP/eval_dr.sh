#!/bin/bash

# == keep unchanged ==
policy_name=DP
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
checkpoint_num=${6}
gpu_id=${7}
exp_name=${8}
eval_distilled_model=${9}
DEBUG=False

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..

if [ ! -z "${exp_name}" ]; then
    exp_name_arg="--exp_name ${exp_name}"
else
    exp_name_arg=""
fi

if [ ! -z "${eval_distilled_model}" ]; then
    eval_distilled_model_arg="--eval_distilled_model"
else
    eval_distilled_model_arg=""
fi

PYTHONWARNINGS=ignore::UserWarning \
python -u script/eval_policy_danger_rate.py --config policy/$policy_name/deploy_policy.yml \
    ${exp_name_arg} \
    ${eval_distilled_model_arg} \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --expert_data_num ${expert_data_num} \
    --checkpoint_num ${checkpoint_num} \
    --seed ${seed} \
    