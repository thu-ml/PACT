if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
import copy
from functools import partial

import tqdm
import pickle
import numpy as np

from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.workspace.robotworkspace import RobotWorkspace, create_dataloader
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

from extend.saveable_energy_function import (
    get_energy_by_task_name, 
    get_env_keys_by_task_name
)
OmegaConf.register_new_resolver("eval", eval, replace=True)

class RobotDistillationWorkspace(RobotWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        
        self.raw_energy_func = get_energy_by_task_name(cfg.task.name)
        
        assert self.model.noise_scheduler.config.prediction_type == "epsilon", \
            "Only epsilon prediction type is supported for distillation"
        assert self.model.obs_as_global_cond, \
            "Only global cond is supported for distillation"    
    
    def run(self):
        cfg = copy.deepcopy(self.cfg)
        seed = cfg.training.seed
        head_camera_type = cfg.head_camera_type

        if cfg.pretrained_ckpt is not None:
            print(f"Resuming from checkpoint {cfg.pretrained_ckpt}")
            original_output_dir = self._output_dir
            
            # Only use pretrained weight, do not inherit global_step and epoch
            # exclude params from optimizer to ensure re-initializing optimizer
            self.load_checkpoint(path=cfg.pretrained_ckpt, exclude_keys=["optimizer"])

            # re-initialize model with ema model when use_ema=True
            if cfg.training.use_ema:
                self.model = copy.deepcopy(self.ema_model)

                # re-initialize optimizer with model parameters
                self.optimizer = hydra.utils.instantiate(
                    cfg.optimizer, params=self.model.parameters())
            else:
                # re-initialize ema model 
                self.ema_model = None

          
            self.global_step = self.epoch = 0
            self._output_dir = original_output_dir
        else:
            raise ValueError("From Scratch training is not supported for distillation")

        # load env meta info for calculate energy
        with open(os.path.join(cfg.distillation.env_meta_path), "rb") as f:
            env_meta = pickle.load(f)
        for key in env_meta:
            if isinstance(env_meta[key], np.ndarray):
                env_meta[key] = torch.from_numpy(env_meta[key]).to(cfg.training.device)
                
        self.energy_func = partial(
            self.raw_energy_func, env_meta=env_meta)
        
        # load reference model for distillation
        self.ref_model = None
        if not cfg.distillation.use_self_distillation:
            self.ref_model = copy.deepcopy(self.model)
            for param in self.ref_model.parameters():
                param.requires_grad = False
            self.ref_model.eval()
        else:
            # TODO: implement self distillation
            raise NotImplementedError("Self distillation is not supported yet")

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # get normalizer from model
        normalizer = self.model.normalizer

        # configure dataset
        env_keys = get_env_keys_by_task_name(cfg.task.name)
        
        cfg.task.dataset.env_keys = env_keys
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = create_dataloader(dataset, **cfg.dataloader)
        # set normalizer to datasetx
        dataset.normalizer = normalizer

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = create_dataloader(val_dataset, **cfg.val_dataloader)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs) //
            cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # configure env
        # env_runner: BaseImageRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseImageRunner)
        # env_runner = None

        # configure logging
        if cfg.use_wandb:
            import wandb
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                }
            )


        # configure checkpoint
        topk_manager = TopKCheckpointManager(save_dir=os.path.join(self.output_dir, "checkpoints"),
                                             **cfg.checkpoint.topk)

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        if self.ref_model is not None:
            self.ref_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")

        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dataset.postprocess(batch, device)
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                        
                        raw_loss = self.model.compute_distillation_loss(
                            batch, 
                            energy_func=self.energy_func,
                            ref_model=self.ref_model, 
                            guidance_iterations=cfg.distillation.guidance_iterations,
                            max_timesteps_for_guidance=cfg.distillation.max_timesteps_for_guidance,
                            min_guidance_scale=cfg.distillation.min_guidance_scale,
                            max_guidance_scale=cfg.distillation.max_guidance_scale,
                        )

                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()
                        
                        # step optimizer
                        if (self.global_step % cfg.training.gradient_accumulate_every == 0):
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(
                            lr=lr_scheduler.get_last_lr()[0],
                            loss=raw_loss_cpu,
                            refresh=False
                        )
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            json_logger.log(step_log)
                            if cfg.use_wandb:
                                wandb_run.log(step_log, step=self.global_step)
                            self.global_step += 1

                        if (cfg.training.max_train_steps
                                is not None) and batch_idx >= (cfg.training.max_train_steps - 1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log["epoch_average_train_loss"] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                # if (self.epoch % cfg.training.rollout_every) == 0:
                #     runner_log = env_runner.run(policy)
                #     # log all
                #     step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dataset.postprocess(batch, device)
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps
                                        is not None) and batch_idx >= (cfg.training.max_val_steps - 1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log["val_loss"] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = train_sampling_batch
                        obs_dict = batch["obs"]
                        gt_action = batch["action"]

                        result = policy.predict_action(obs_dict)
                        pred_action = result["action_pred"]
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log["train_action_mse_error"] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                # checkpoint
                if ((self.epoch + 1) % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    save_name = pathlib.Path(self.cfg.task.dataset.zarr_path).stem
                    self.save_checkpoint(f"checkpoints/{save_name}-{seed}/{self.cfg.exp_name}/{self.epoch + 1}.ckpt", exclude_keys=["ref_model"])

                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                json_logger.log(step_log)
                if cfg.use_wandb:
                    wandb_run.log(step_log, step=self.global_step)
                self.global_step += 1
                self.epoch += 1

        if cfg.use_wandb:
            wandb_run.finish()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = RobotDistillationWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
