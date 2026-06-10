import numpy as np
import torch

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (SequenceSampler, downsample_mask,
                                             get_val_mask)
from diffusion_policy.dataset.robot_image_dataset import RobotImageDataset


class RobotImageDistillationDataset(RobotImageDataset):

    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        batch_size=128,
        max_train_episodes=None,
        env_keys=None,
    ):
        super(RobotImageDataset, self).__init__()
        # print("In dataset class", zarr_path)
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            # keys=['head_camera', 'front_camera', 'left_camera', 'right_camera', 'state', 'action'],
            keys=["head_camera", "state", "action"] + env_keys,
        )
        self.env_keys = env_keys

        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.batch_size = batch_size
        sequence_length = self.sampler.sequence_length
        self.buffers = {
            k: np.zeros((batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()
        }
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}
        for v in self.buffers_torch.values():
            v.pin_memory()

    def get_normalizer(self, mode="limits", **kwargs):
        print("Normalizer is re-calculated,"
              " which should be avoided when continuing training with distillation")
        return super().get_normalizer(mode=mode, **kwargs)
        
    def _sample_to_data(self, sample):
        raise NotImplementedError(
            "This method should not be called for distillation dataset")

    def postprocess(self, samples, device):
        processed_samples = super().postprocess(samples, device)
        
        env_info = {
            k: samples[k].to(device, non_blocking=True) 
            for k in self.env_keys
        }
        processed_samples["env"] = env_info
        return processed_samples