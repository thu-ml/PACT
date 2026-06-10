import torch
import numpy as np
from collections import deque

from diffusion_policy.env_runner.dp_runner import DPRunner
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class DPBatchRunner(DPRunner):

    def __init__(
        self, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

    def stack_last_n_obs(self, all_obs, n_steps):
        assert len(all_obs) > 0
        all_obs = list(all_obs)
        bsz = all_obs[0].shape
        if isinstance(all_obs[0], np.ndarray):
            result = np.zeros((bsz, n_steps) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[:, start_idx:] = np.array(all_obs[:, start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:, :start_idx] = result[:, start_idx]
        elif isinstance(all_obs[0], torch.Tensor):
            result = torch.zeros((bsz, n_steps) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[:, start_idx:] = torch.stack(all_obs[:, start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:, :start_idx] = result[:, start_idx]
        else:
            raise RuntimeError(f"Unsupported obs type {type(all_obs[0])}")
        return result

    def get_action(self, policy: BaseImagePolicy, observaton=None):
        device, dtype = policy.device, policy.dtype
        if observaton is not None:
            self.obs.append(observaton)  # update
        obs = self.get_n_steps_obs()

        # create obs dict
        np_obs_dict = dict(obs)
        # device transfer
        obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
        # run policy
        with torch.no_grad():
            action_dict = policy.predict_action(obs_dict)

        # device_transfer
        np_action_dict = dict_apply(action_dict, lambda x: x.detach().to("cpu").numpy())
        actions = np_action_dict["action"][:self.n_action_steps]
        return actions

    def get_action_with_guid(self, policy: BaseImagePolicy, TASK_ENV, observaton=None, record_denoising_process=False):
        raise NotImplementedError("get_action_with_guid is not implemented for DPBatchRunner")