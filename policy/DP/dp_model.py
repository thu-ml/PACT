import numpy as np
import torch
import hydra
import dill
import sys, os

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
sys.path.append(parent_dir)

from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from diffusion_policy.env_runner.dp_runner import DPRunner

class DP:

    def __init__(self, ckpt_file: str, n_obs_steps, n_action_steps, use_ema=None):
        self.policy = self.get_policy(ckpt_file, None, "cuda:0", use_ema)
        self.runner = DPRunner(n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)

    def update_obs(self, observation):
        self.runner.update_obs(observation)
    
    def reset_obs(self):
        self.runner.reset_obs()

    def get_action(self, observation=None):
        action = self.runner.get_action(self.policy, observation)
        return action
    
    def get_action_with_guid(self, TASK_ENV, observation=None, record_denoising_process=False):
        action = self.runner.get_action_with_guid(self.policy, TASK_ENV, observation, record_denoising_process)
        return action

    def get_last_obs(self):
        return self.runner.obs[-1]

    def get_policy(self, checkpoint, output_dir, device, use_ema=None):
        # load checkpoint
        payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
        cfg = payload["cfg"]

        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=output_dir)
        workspace: RobotWorkspace
        workspace.load_payload(
            payload, exclude_keys=["old_model", "ref_model", "optimizer"], include_keys=None)

        policy = workspace.model
        if use_ema is None:
            use_ema = cfg.training.use_ema

        if use_ema:
            policy = workspace.ema_model
        elif "old_model" in payload["state_dicts"]:
            print("Using old model for on policy distillation")
            policy.load_state_dict(payload["state_dicts"]["old_model"])
        

        device = torch.device(device)
        policy.to(device)
        policy.eval()

        return policy
