<div align="center">
  <h1 style="font-size: 4rem; font-weight: bold; color: #667eea; margin: 20px 0; display: flex; align-items: center; justify-content: center; gap: 20px; border-bottom: none; padding-bottom: 5;">
    PACT: Self-Evolving Physical Safety Alignment for Diffusion Policies in Embodied Manipulation
  </h1>
</div>
<div align="center" style="line-height: 1;">
  <a href="https://ethan-iai.github.io/pact/"><img alt="Homepage"
    src="https://img.shields.io/badge/PACT-Homepage-4287f5?logo=probot&logoColor=white"/></a>
  <a href="https://arxiv.org/abs/2606.08414"><img alt="Paper"
    src="https://img.shields.io/badge/arXiv-Paper-B31B1B?logo=arxiv"/></a>
  <a href="https://huggingface.co/datasets/Ethan-pooh/pact"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-PACT-ffc107?color=ffc107&logoColor=white"/></a>
</div>


## Overview

PACT is a **self-evolving post-training framework** for aligning diffusion policies with physical safety constraints in embodied manipulation. Starting from pretrained diffusion policies, PACT uses **self-rollouts** and automatically computed **physical constraints** to distill constraint gradients into the policy, improving safety **without requiring demonstrations, task rewards, interventions, or outcome annotations**.

This repository contains the code for running PACT on [RoboTwin](https://robotwin-platform.github.io)-based manipulation tasks, including base-policy evaluation, on-policy distillation, and post-training policy evaluation.

## Updates

- 🔥 [Jun 2026] We released the [project website](https://ethan-iai.github.io/pact/), [arXiv paper](https://arxiv.org/abs/2606.08414), and required [data and pre-trained policy checkpoints](https://huggingface.co/datasets/Ethan-pooh/pact) on Hugging Face.


## Table of Contents

- [Installation](#installation)
- [Model Checkpoints & Data](#model-checkpoints-and-data)
- [Post-Training & Evaluation](#evaluation-and-post-training)
- [Citation](#citation)


## Installation

We recommend using a Linux system with NVIDIA GPUs. Our experiments were conducted on RTX x090 GPUs (24 GB VRAM). For Diffusion Policy (DP) training, we use a per-GPU batch size of 128, demonstrating that the method is relatively memory efficient and can be trained comfortably within a budget less than 24 GB VRAM.

Clone this repository and create a conda environment:

```bash
conda create -n pact python=3.10 -y
conda activate pact
```

Install RoboTwin and download RoboTwin assets:

```bash
bash script/_install.sh
bash script/_download_assets.sh
```

PACT modifies several assets used by custom tasks such as `handover_apple` and `pour_water_to_cup`. Replace the downloaded assets with the PACT versions:

```bash
bash script/_replace_assets.sh
```

Install PACT dependencies:

```bash
pip install -r requirements.txt
```

Install base policy dependencies:

```bash
cd policy/DP
pip install -e .
cd ../..
```

Create the symbolic link to cost-function utilities:

```bash
ln -s ../../extend ./policy/DP/extend
```

## Model Checkpoints and Data

Download the pretrained base policy checkpoints from [Hugging Face](https://huggingface.co/datasets/Ethan-pooh/pact), and organize them as follows:

```bash
# generally, the structure is as follows:
PACT/policy/DP/checkpoints/
├── ${TASK_NAME}-demo_randomized-200-0/600.ckpt

# concretely, the arranged structure containing all the released checkpoints is as follows:
PACT/policy/DP/checkpoints/
├── handover_apple-demo_randomized-200-0/600.ckpt
├── handover_block-demo_randomized-200-0/600.ckpt
├── pick_diverse_bottles-demo_randomized-200-0/600.ckpt
├── pick_dual_bottles-demo_randomized-200-0/600.ckpt
├── place_dual_shoes-demo_randomized-200-0/600.ckpt
├── pour_water_to_cup-demo_randomized-200-0/600.ckpt
└── stack_blocks_two-demo_randomized-200-0/600.ckpt
```

Download the pre-generated instruction dataset `data.tar.gz` from [Hugging Face](https://huggingface.co/datasets/Ethan-pooh/pact), and organize it as follows:

```bash
# generally, the structure is as follows:
PACT/data/
├── data/${TASK_NAME}/demo_randomized/instructions
├── env_meta.pkl    # meta environment infomation shared by all tasks

# concretely, the arranged structure containing all the released instruction datasets is as follows:
PACT/data/
├── data/handover_apple/demo_randomized/instructions
├── data/handover_block/demo_randomized/instructions
├── data/pick_diverse_bottles/demo_randomized/instructions
├── data/pick_dual_bottles/demo_randomized/instructions
├── data/place_dual_shoes/demo_randomized/instructions
├── data/pour_water_to_cup/demo_randomized/instructions
├── data/stack_blocks_two/demo_randomized/instructions
└── env_meta.pkl    # meta environment infomation shared by all tasks
```

## Evaluation and Post-Training

### Evaluate a Base Policy

Run evaluation from `policy/DP`:

```bash
cd policy/DP

# bash eval_dr.sh ${TASK_NAME} ${TASK_CONFIG} ${CKPT_CONFIG} ${EPISODE_NUM} ${SEED} ${CKPT_NUM} ${GPU_ID}
bash eval_dr.sh pick_dual_bottles demo_randomized demo_randomized 200 0 600 0
```

### Run PACT Post-Training

Run on-policy distillation from the repository root:

```bash
# bash policy/DP/on_policy_distill_multigpu.sh ${TASK_NAME} ${TASK_CONFIG} ${BASE_EPISODE_NUM} ${SEED} ${ACTION}
# Train with four specified GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 bash policy/DP/on_policy_distill_multigpu.sh pick_dual_bottles onpolicy_randomized 200 0 14
```

### Evaluate a PACT Post-Trained Policy

Run evaluation from `policy/DP`:

```bash
cd policy/DP

# bash eval_dr.sh ${TASK_NAME} ${TASK_CONFIG} ${CKPT_CONFIG} ${EPISODE_NUM} ${SEED} ${CKPT_NUM} ${GPU_ID} ${CKPT_PATH} ${EVAL_DISTILLATION}
bash eval_dr.sh pick_dual_bottles demo_randomized onpolicy_randomized 200 0 20 0 playground/dp/pick_dual_bottles/exptime_robot_on_policy_distillation_pick_dual_bottles True
```

## Acknowledgments

We gratefully acknowledge the authors of [RoboTwin](https://robotwin-platform.github.io) and [Diffusion Policy](https://github.com/real-stanford/diffusion_policy). PACT builds on their excellent simulation benchmark and policy implementation.

## Citation

If you find our work helpful, please cite us:

```bibtex
@article{wu2026pact,
  title={PACT: Self-Evolving Physical Safety Alignment for Diffusion Policies in Embodied Manipulation}, 
  author={Wu, Lingxuan and Zhu, Zijian and Wang, Lizhong and Ying, Chengyang and Chen, Huayu and Yang, Xiao and Liu, Fangming and Zhu, Jun},
  journal={arXiv preprint arXiv:2606.08414},
  year={2026},
}
```