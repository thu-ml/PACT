#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
action_dim=${5}

head_camera_type=D435

DEBUG=False
save_ckpt=True

alg_name=on_policy_robot_dp_distillation_multigpu_$action_dim
config_name=${alg_name}
exp_name=${task_name}-${alg_name}

# for resume
# run_dir="/data/wulingxuan/PACT/playground/dp/handover_apple/2026.06.10-09.51.36_robot_on_policy_distillation_handover_apple"


ABLATION=False
if [ $ABLATION = True ]; then
    echo -e "\033[33mAblation mode! setting teacher_model_base=ref\033[0m"
    ablation_args="training.teacher_model_base=ref"
else
    ablation_args=""
fi

if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

export HYDRA_FULL_ERROR=1 

# Set NCCL timeout to avoid timeout during long rollout waits
# Default is 30 minutes (1800s), set to 2 hours (7200s) for safety
# export NCCL_TIMEOUT=7200

# Optional: Set other NCCL environment variables for better stability
# export NCCL_DEBUG=INFO  # Uncomment for debugging NCCL issues
# export NCCL_IB_DISABLE=1  # Disable InfiniBand if having issues
# export NCCL_SOCKET_IFNAME=eth0  # Specify network interface if needed

base_seed=0
base_task_config=demo_randomized
base_expert_data_num=200
base_epoch=600

export RUN_TIME=$(date +"%Y.%m.%d-%H.%M.%S")

if [ -d "$run_dir" ]; then
    echo -e "\033[32mResuming distillationfrom run_dir: $run_dir\033[0m"
    accelerate launch policy/DP/train.py --config-name=${config_name}.yaml \
        hydra.run.dir=${run_dir} \
        hydra.sweep.dir=${run_dir} \
        rollout.env.task_name=${task_name} \
        logging.project=${exp_name} \
        distillation.env_meta_path="data/env_meta.pkl" \
        pretrained_ckpt="policy/DP/checkpoints/${task_name}-${base_task_config}-${base_expert_data_num}-${base_seed}/${base_epoch}.ckpt" \
        task.name=${task_name} \
        task.dataset.zarr_path="data/${task_name}-${task_config}-${expert_data_num}.zarr" \
        training.debug=$DEBUG \
        training.seed=${seed} \
        ${ablation_args} \
        exp_name=${exp_name} \
        logging.mode=${wandb_mode} \
        setting=${task_config} \
        expert_data_num=${expert_data_num} \
        head_camera_type=$head_camera_type
else
    echo -e "\033[31mTrain from pretrained checkpoint\033[0m"
    accelerate launch policy/DP/train.py --config-name=${config_name}.yaml \
        run_time=${RUN_TIME} \
        rollout.env.task_name=${task_name} \
        logging.project=${exp_name} \
        distillation.env_meta_path="data/env_meta.pkl" \
        pretrained_ckpt="policy/DP/checkpoints/${task_name}-${base_task_config}-${base_expert_data_num}-${base_seed}/${base_epoch}.ckpt" \
        task.name=${task_name} \
        task.dataset.zarr_path="data/${task_name}-${task_config}-${expert_data_num}.zarr" \
        training.debug=$DEBUG \
        training.seed=${seed} \
        ${ablation_args} \
        exp_name=${exp_name} \
        logging.mode=${wandb_mode} \
        setting=${task_config} \
        expert_data_num=${expert_data_num} \
        head_camera_type=$head_camera_type
fi


