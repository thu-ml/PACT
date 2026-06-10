from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

class DiffusionUnetImagePolicy(BaseImagePolicy):

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        obs_encoder: MultiImageObsEncoder,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(model_output, t, trajectory, generator=generator, **kwargs).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert "past_action" not in obs_dict  # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs,
        )

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {"action": action, "action_pred": action_pred}
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_guided_model_output(
        self, samples, timesteps, model_outputs, 
        env_states, energy_func,
        max_timesteps_for_guidance=3,
        guidance_scales=0.1, 
        guidance_iterations=1,
    ):
        guided_model_outputs = model_outputs.clone().detach()
        for batch_idx in range(samples.shape[0]):
            t = timesteps[batch_idx]
            if (
                max_timesteps_for_guidance is not None 
                and t >= max_timesteps_for_guidance
            ):
                continue
            
            env_state = {k: v[batch_idx] for k, v in env_states.items()}
            sample = samples[batch_idx: batch_idx + 1]  # (B, ...) -> (1, ...)
            model_output = model_outputs[batch_idx: batch_idx + 1]  # (B, ...) -> (1, ...)
            mpgd_w = guidance_scales[batch_idx] \
                if not isinstance(guidance_scales, float) else guidance_scales
            
            sample_req = sample.clone().detach()
            for _ in range(guidance_iterations):
                sample_req.requires_grad_(True)
                sqrt_alpha_cumprod_t, sigma_t = \
                    get_alpha_sigma_from_scheduler(self.noise_scheduler, t, sample_req)
                sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.to(sample.device)
                sigma_t = sigma_t.to(sample.device)
            
                # get x_0:t from x_t and eps_t
                guided_sample_0_pred_ori = \
                    (sample_req - sigma_t * model_output) / sqrt_alpha_cumprod_t
                if self.noise_scheduler.clip_sample:
                    guided_sample_0_pred_ori = guided_sample_0_pred_ori.clamp(-1.0, 1.0)
            
                trajectory_unnormed = self.normalizer["action"].unnormalize(
                    guided_sample_0_pred_ori)[..., : self.action_dim]
                
                energy, _ = energy_func(
                    env_state=env_state,
                    trajectory_unnormed=trajectory_unnormed,
                    device=sample.device
                )
                grad = torch.autograd.grad(
                    energy, sample_req,
                    retain_graph=False, 
                    create_graph=False
                )[0]
                sample_req = sample_req.detach() - grad * mpgd_w
            
            sample_delta = sample_req - sample
            
            beta_t = self.noise_scheduler.betas[t]
            guided_model_output = model_output - (sigma_t / beta_t) * sample_delta
            
            guided_model_outputs[batch_idx: batch_idx + 1] = guided_model_output
        
        guided_model_outputs = guided_model_outputs.detach()
        
        return guided_model_outputs
    
    def compute_distillation_loss(
        self, batch, 
        energy_func=None,
        ref_model=None,
        guidance_iterations=2,
        min_guidance_scale=0.01,
        max_guidance_scale=0.15,
        max_timesteps_for_guidance=3,
    ):
        env_state = batch["env"]
        batch.pop("env")
        
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        
        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            raise NotImplementedError("Only global cond is supported for distillation")
        
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        if batch.get("timesteps", None) is None:
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (bsz, ),
                device=trajectory.device,
            ).long()
        else:
            timesteps = batch["timesteps"].to(trajectory.device)
        
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # compute guidance scale
        guidance_scales = torch.rand(size=(batch_size,), device=trajectory.device) \
            * (max_guidance_scale - min_guidance_scale) + min_guidance_scale
        
        with torch.no_grad():
            ref_local_cond = None
            ref_global_cond = None
            if ref_model is None:
                assert False, "ref_model should not be None for distillation"
                # ref_model = self
                # ref_global_cond = global_cond.clone().detach()
            else:
                if self.obs_as_global_cond:
                    ref_nobs_features = ref_model.obs_encoder(this_nobs)
                    # reshape back to B, Do
                    ref_global_cond = ref_nobs_features.reshape(batch_size, -1)
                else:
                    raise NotImplementedError("Only global cond is supported for distillation")
            
            ref_pred = ref_model.model(
                noisy_trajectory, timesteps, 
                local_cond=ref_local_cond, 
                global_cond=ref_global_cond
            )
        
        # calculate guided score here
        # only use current envrionment state to calculate guidance
        # same as guidance sampling
        for k in env_state:
            env_state[k] = env_state[k][:, self.n_obs_steps - 1] # (B, T, ...) -> (B, ...)
        guided_pred = self.compute_guided_model_output(
            noisy_trajectory, timesteps, ref_pred, 
            env_state, energy_func, 
            max_timesteps_for_guidance=max_timesteps_for_guidance,
            guidance_scales=guidance_scales, 
            guidance_iterations=guidance_iterations,
        )
        
        loss_terms = {}
        pred = self.model(
            noisy_trajectory, timesteps, 
            local_cond=local_cond, 
            global_cond=global_cond
        )
        
        with torch.no_grad():
            loss_terms["old_deviate"] = torch.mean((pred - guided_pred) ** 2).detach()
            loss_terms["old_deviate_max"] = torch.max((pred - guided_pred) ** 2).detach()
        
        loss = F.mse_loss(pred, guided_pred, reduction="mean")
        return loss, loss_terms
    
    def forward(self, *args, **kwargs):
        return self.compute_distillation_loss(*args, **kwargs)
    
    def compute_distillation_loss_with_model_output(self, batch):
        nobs = self.normalizer.normalize(batch["obs"])
        batch_size = batch["model_input"].shape[0]
        
        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            raise NotImplementedError("Only global cond is supported for distillation")

        noisy_trajectory = batch["model_input"]
        timesteps = batch["t"]
        
        pred = self.model(
            noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond)
        
        loss = F.mse_loss(pred, batch["model_output"], reduction="mean")
        return loss

    def compute_loss(self, batch, energy_func=None, temperature=1.0, batch_normalization=False):
        # normalize input
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, ),
            device=trajectory.device,
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        weight = None
        if energy_func is not None:
            with torch.no_grad():
                env_state = batch["env"]
                # calculate guided score here
                # only use current envrionment state to calculate guidance
                # same as guidance sampling
                for k in env_state:
                    env_state[k] = env_state[k][:, self.n_obs_steps - 1] # (B, T, ...) -> (B, ...)
                
                batch_energy = torch.zeros(bsz, device=trajectory.device)
                for batch_idx in range(bsz):
                    energy, _ = energy_func(
                        env_state=env_state[batch_idx],
                        trajectory_unnormed=batch["action"][batch_idx],
                        device=trajectory.device
                    )
                    batch_energy[batch_idx] = energy

                if batch_normalization:
                    batch_energy = batch_energy - batch_energy.mean()
                    batch_energy = batch_energy / (batch_energy.std() + 1e-4)
                    
                weight = torch.exp(- batch_energy / (temperature + 1e-6))
                
        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        if weight is not None:
            loss = loss * weight.unsqueeze(-1)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss
    
def get_alpha_sigma_from_scheduler(scheduler, t, trajectory):
    alpha_prod_t = scheduler.alphas_cumprod[t]
    alpha_t = alpha_prod_t.sqrt()
    sigma_t = (1.0 - alpha_prod_t).sqrt()
    while alpha_t.ndim < trajectory.ndim:
        alpha_t = alpha_t.view(alpha_t.shape + (1,))
        sigma_t = sigma_t.view(sigma_t.shape + (1,))
    return alpha_t, sigma_t
