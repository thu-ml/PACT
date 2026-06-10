if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import gc
import json
import logging
import os
import pathlib
import pickle
import random
import re
import sys
import time
from datetime import timedelta
from functools import partial
import signal
import hydra
import numpy as np
import torch
import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from omegaconf import OmegaConf

from diffusion_policy.common.json_logger import JsonLogger
# from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.workspace.robot_distillation_workspace import \
    RobotDistillationWorkspace
from diffusion_policy.workspace.robotworkspace import create_dataloader

from rollout_utils import compute_success_rate, process_data
from extend.saveable_energy_function import get_env_keys_by_task_name
from extend.seed_utils import load_seeds, split_seeds
from extend.energy_function import get_guid_max_timesteps_by_policy_name, get_guid_param_by_task_name



logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

OmegaConf.register_new_resolver("eval", eval, replace=True)



MAX_STEPS_PER_TASK = {
    "place_dual_shoes": 500,
    "pour_water_to_cup": 300,
    "handover_apple": 270, # Max in eval
    "click_bell": 200,
    "move_can_pot": 200,
    "robotwin2_place_phone_stand": 200,
    "place_a2b_left": 200,
    "place_a2b_right": 200,
    "handover_mic": 200,
    "pick_dual_bottles": 126,   # original 100 steps  # Max in eval
    "pick_diverse_bottles": 136,
    "lift_pot": 200,
    "put_bottles_dustbin": 800,
    "stack_blocks_two": 400,
    "stack_bowls_two": 400,
    "handover_block": 400,
    "place_empty_cup": 200,
    "shake_bottle": 75,
    "move_stapler_pad": 200,
    "place_container_plate": 150,
    "blocks_ranking_rgb": 600,
    "beat_block_hammer": 200,
    "place_mouse_pad": 200,
    "place_shoe": 250,
    "move_pillbottle_pad": 200,
}

def return_decay(step, decay_type):
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001 / 2     # original 0.001
        uphold = 0.5
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.999
    elif decay_type == 3:
        return max(0.75, 1 - 5e-6 * step)
    else:
        assert False

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)


def kill_all_pids(running_pid_set):
    """Kill all processes in running_pid_set
    
    Args:
        running_pid_set (set): Set of process IDs to kill
    """
    print(f"\nKilling all processes in {running_pid_set}")
    for pid_str in running_pid_set:
        try:
            pid = int(pid_str.strip())
            os.kill(pid, signal.SIGTERM)
            print(f"Sent termination signal to process {pid}")
        except (ValueError, ProcessLookupError, OSError) as e:
            print(f"Error killing process {pid_str}: {e}")


def generate_folder_tree_simple(root_dir, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        
        def build_tree(current_path, indent=''):
            try:
                items = sorted(os.listdir(current_path))
                
                for i, item in enumerate(items):
                    item_path = os.path.join(current_path, item)
                    is_last = (i == len(items) - 1)
                    
                    prefix = '--' if is_last else '--'
                    
                    if os.path.isdir(item_path):
                        f.write(f"{indent}{prefix}/{item}\n")
                        build_tree(item_path, indent + ('  ' if is_last else '  '))
                    else:
                        f.write(f"{indent}{prefix}{item}\n")
                        
            except Exception as e:
                f.write(f"{indent}[Error: {str(e)}]\n")
        
        root_name = os.path.basename(root_dir) if os.path.basename(root_dir) else root_dir
        f.write(f"/{root_name}\n")
        build_tree(root_dir)



class RobotOnPolicyDistillationWorkspace(RobotDistillationWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
    
    def get_checkpoint_path(self, tag="latest"):
        checkpoints_dir = pathlib.Path(self.output_dir).joinpath("checkpoints")
        
        if tag == "latest":
            if not checkpoints_dir.exists():
                return None
            pattern = re.compile(r'^(\d+)\.ckpt$')
            max_epoch = -1
            latest_ckpt = None
            
            for ckpt_file in checkpoints_dir.glob("*.ckpt"):
                match = pattern.match(ckpt_file.name)
                if match:
                    epoch = int(match.group(1))
                    if epoch > max_epoch:
                        max_epoch = epoch
                        latest_ckpt = ckpt_file
            return latest_ckpt
        else:
            return checkpoints_dir / f"{tag}.ckpt"
    
    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        # get guidance parameters
        MPGD_N, MPGD_W = get_guid_param_by_task_name(cfg.rollout.env.task_name, policy_name="DP")
        max_timesteps_for_guidance = get_guid_max_timesteps_by_policy_name(policy_name="DP")   
        
        # set guidance-related hyperparameters for logging
        cfg.distillation.min_guidance_scale = MPGD_W
        cfg.distillation.max_guidance_scale = MPGD_W
        cfg.distillation.guidance_iterations = MPGD_N
        cfg.distillation.max_timesteps_for_guidance = max_timesteps_for_guidance

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # Set the timeout to, for example, `cfg.training.timeout` seconds ()
        timeout_value = timedelta(seconds=cfg.training.timeout) 
        init_kwargs = InitProcessGroupKwargs(timeout=timeout_value)
        accelerator = Accelerator(
            log_with='wandb', kwargs_handlers=[ddp_kwargs, init_kwargs])
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop('project')
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )
        device = accelerator.device
        
        if cfg.pretrained_ckpt is not None:
            logger.info(f"Resuming from checkpoint {cfg.pretrained_ckpt}")
            original_output_dir = self._output_dir
            
            # Only use pretrained weight, do not inherit global_step and epoch
            # exclude params from optimizer to ensure re-initializing optimizer
            self.load_checkpoint(path=cfg.pretrained_ckpt, exclude_keys=["optimizer"])

            # re-initialize model with ema model when use_ema=True
            if cfg.training.use_ema:
                with torch.no_grad():
                    # copy ema weight to model
                    for param, ema_param in zip(
                        self.model.parameters(), self.ema_model.parameters(), strict=True):
                        param.data.copy_(ema_param.detach().clone().data)
                
                last_ckpt_path = cfg.pretrained_ckpt
            else:
                # re-initialize ema model 
                self.ema_model = None

            self.epoch = self.global_step = 0
            self._output_dir = original_output_dir
        else:
            if not cfg.training.resume:
                raise ValueError("From Scratch training is not supported for on-policy distillation")

        # load env meta info for calculate energy
        with open(os.path.join(cfg.distillation.env_meta_path), "rb") as f:
            env_meta = pickle.load(f)
        for key in env_meta:
            if isinstance(env_meta[key], np.ndarray):
                env_meta[key] = torch.from_numpy(env_meta[key]).to(device)
                
        self.energy_func = partial(
            self.raw_energy_func, env_meta=env_meta)
        
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()
        for param in self.old_model.parameters():
            param.requires_grad = False
        
        # load reference model for kl-regularization and logging
        self.ref_model = None
        if cfg.training.teacher_model_base == "ref":
            logger.info(f"Using ref model as teacher model")
            self.ref_model = copy.deepcopy(self.model)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
        
        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path is not None and lastest_ckpt_path.is_file():
                logger.info(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                self.epoch += 1 # start next epoch
                self.global_step += 1   # start next global step
                logger.info(f"Continue training from epoch {self.epoch} (global_step {self.global_step})")
                last_ckpt_path = str(lastest_ckpt_path)

        # configure dataset
        env_keys = get_env_keys_by_task_name(cfg.task.name)
        cfg.task.dataset.env_keys = env_keys
        
        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        self.model, self.optimizer = accelerator.prepare(
            self.model, self.optimizer)

        # device transfer
        for model in [self.ema_model, self.ref_model, self.old_model]:
            if model is not None:
                model.to(device)
        
        if cfg.training.debug:
            collect_num = 1
            cfg.rollout.num_batches_per_epoch = collect_num
            cfg.rollout.env.collect_num_per_worker = collect_num
            cfg.rollout.env.collect_num = collect_num
            
            cfg.training.num_epochs = 4
            cfg.training.num_inner_epochs = 100
            cfg.training.max_train_steps = 100
            cfg.training.max_val_steps = 3
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            
            # to avoid drop-out
            batch_size = 64
            cfg.task.dataset.batch_size = batch_size
            cfg.dataloader.batch_size = batch_size

        # set up rollout environment configuration
        env_cfg = cfg.rollout.env

        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)

        if visible is not None:
            cfg.rollout.gpus = [int(x) for x in visible.split(",")]
        else:
            cfg.rollout.gpus = list(range(torch.cuda.device_count()))
        
        # fetch variables from env_cfg
        num_gpus = len(cfg.rollout.gpus)
        worker_per_gpu = env_cfg.worker_per_gpu
        collect_num_per_worker = env_cfg.collect_num_per_worker
        max_seed_num = env_cfg.max_seed_num
        task_num = num_gpus * worker_per_gpu
        if env_cfg.collect_max_steps is None:
            env_cfg.collect_max_steps = MAX_STEPS_PER_TASK[env_cfg.task_name]
        
        # load seed list
        seed_list = load_seeds(cfg.rollout.seed_list)
        seed_list = seed_list[:max_seed_num]

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            first_epoch = self.epoch
            for _ in range(first_epoch, cfg.training.num_epochs):
                # set epoch specific variables
                save_path = os.path.join(
                    self.output_dir,
                    "rollouts", 
                    f'epoch-{self.epoch}'
                )
                zarr_path = os.path.join(save_path, 'data.zarr')
                
                # Synchronize all processes before rollout to avoid NCCL timeout
                accelerator.wait_for_everyone()

                rollout_done_flag = os.path.join(save_path, 'logs', "rollout_done.flag")
                
                if os.path.exists(zarr_path):
                    logger.info(f"Skipping rollout for epoch {self.epoch}" 
                                " because zarr file already exists")
                    
                # launch rollouts for this epoch
                if accelerator.is_main_process and not os.path.exists(zarr_path):
                    # shuffle seeds before each rollout
                    random.shuffle(seed_list)
                    seed_list_per_task = split_seeds(
                        seed_list, task_num, verbose=cfg.training.debug)
                    
                    # start rollout 
                    running_pid_set = set()
                    
                    stdout_path = os.path.join(save_path, 'stdouts')
                    rollout_log_path = os.path.join(save_path, 'logs')
                    env_cfg_path = os.path.join(save_path, "cfgs")
                    
                    for path in [rollout_log_path, env_cfg_path, stdout_path]:
                        os.makedirs(path, exist_ok=True)
                    
                    env_cfg["save_path"] = save_path
                    env_cfg["ckpt_path"] = last_ckpt_path       
                    # only use ema model when the model has not been trained yet
                    env_cfg["use_ema"] = (cfg.training.use_ema and self.epoch == 0) 
                    
                    system_cmd_list = []
                    for gpu_idx, gpu in enumerate(cfg.rollout.gpus):
                        for worker_idx in range(worker_per_gpu):
                            task_idx = gpu_idx * worker_per_gpu +  worker_idx
                            env_cfg["seed_list"] = seed_list_per_task[task_idx]
                            random.shuffle(env_cfg["seed_list"])
                            env_cfg["task_idx"] = task_idx

                            with open(os.path.join(env_cfg_path, f'{task_idx}.json'), "w", encoding="utf-8") as fp:
                                json.dump(OmegaConf.to_container(env_cfg, resolve=True), fp, indent=4)
                            
                            stdout_file_path = os.path.join(stdout_path, f'{task_idx}.out')
                            system_cmd = \
                                f"CUDA_VISIBLE_DEVICES={gpu} nohup bash worker.sh {env_cfg_path} {task_idx} > {stdout_file_path} 2>&1 &"
                            system_cmd_list.append(system_cmd)
                            os.system(system_cmd)

                    # print(f"Waiting for all rollout processes to finish")
                    finish_signal = [False] * task_num
                    try:
                        while True:
                            time.sleep(1)
                            for task_idx in range(task_num):
                                stdout_file_path = os.path.join(stdout_path, f'{task_idx}.out')
                                if os.path.exists(stdout_file_path):
                                    with open(stdout_file_path, 'r') as fp:
                                        lines_to_read = 5
                                        for _ in range(lines_to_read):
                                            line = fp.readline()
                                            pid_match = re.search(r'\d+', line)
                                            if pid_match:
                                                pid = pid_match.group()
                                                running_pid_set.add(pid)
                                                break

                                if os.path.exists(os.path.join(rollout_log_path, f'signal_{task_idx}')):
                                    finish_signal[task_idx] = True
                                    running_pid_set.discard(pid)
                                    
                            hdf5_data_dir = os.path.join(save_path, 'data')
                            if os.path.exists(hdf5_data_dir):
                                total_count = len(os.listdir(hdf5_data_dir))
                                print(f"Data collection progress: {total_count}/{collect_num_per_worker * task_num},"
                                    f" remaining parallel processes: {len(running_pid_set)}", end='\r', flush=True)
                                
                            if all(finish_signal):
                                print(f"Data collection progress: {total_count}/{collect_num_per_worker * task_num},"
                                    f" remaining parallel processes: {len(running_pid_set)}")
                                logger.info("All processes finished")
                                break
                            
                    except KeyboardInterrupt:
                        logger.info("\nUser interrupted, cleaning up...")
                        kill_all_pids(running_pid_set)
                        sys.exit(1)

                    rollout_success_rate = compute_success_rate(save_path)
                    accelerator.log({
                        "rollout_success_rate": rollout_success_rate,
                        "epoch": self.epoch,
                    }, step=self.global_step)
                
                    logger.info(f"Processing data from {save_path} to {zarr_path}...")
                    process_data(save_path, zarr_path)
                    logger.info(f"Zarr is dumped to {zarr_path} successfully")

                    with open(rollout_done_flag, "w") as f:
                        f.write("done\n")
                
                if not accelerator.is_main_process:
                    while not os.path.exists(rollout_done_flag):
                        time.sleep(1.0)

                # Synchronize all processes after rollout completion
                accelerator.wait_for_everyone()
                
                gc.collect()
                torch.cuda.empty_cache()
                
                dataset = None
                max_retries = 3
                for retry_idx in range(max_retries):
                    if dataset is not None:
                        logger.info(
                            f"[rank {accelerator.process_index}] Dataset instantiated successfully"
                            f" on try {retry_idx - 1}/{max_retries}")
                        break

                    try:
                        if cfg.training.debug:
                            generate_folder_tree_simple(
                                zarr_path,
                                f"rollout_{self.epoch}_rank_{accelerator.process_index}_try_{retry_idx}.txt"
                            )
                        dataset = hydra.utils.instantiate(
                            cfg.task.dataset,
                            zarr_path=zarr_path
                        )
                    except Exception as e:
                        logger.info(f"[rank {accelerator.process_index}] Error instantiating dataset: {e}")
                        print(f"[rank {accelerator.process_index}] Error instantiating dataset: {e}")
                        time.sleep(1)
                    
                        logger.info(
                            f"[rank {accelerator.process_index}] Retrying to instantiate dataset"
                            f" (retry {retry_idx + 1}/{max_retries})...")
                        
                dataloader = create_dataloader(dataset, **cfg.dataloader)
                dataloader = accelerator.prepare(dataloader)
                
                # save batch for sampling
                train_sampling_batch = None
                
                step_log = dict()
                train_losses = list()
                
                # ========= train for this epoch ==========
                self.model.train()
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                for inner_epoch in range(cfg.training.num_inner_epochs):
                    # if use ddp, should set epoch_idx here
                    with tqdm.tqdm(
                        dataloader,
                        desc=f"Training epoch {self.epoch}.{inner_epoch}",
                        leave=False,
                        disable=not accelerator.is_main_process,
                        mininterval=cfg.training.tqdm_interval_sec,
                    ) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dataset.postprocess(batch, device)
                            if train_sampling_batch is None:
                                train_sampling_batch = batch
                            
                            teacher_model = self.ref_model \
                                if cfg.training.teacher_model_base == "ref" \
                                else self.old_model
                            
                            # NOTE: forward <-> compute_distillation_loss
                            raw_loss, loss_terms = self.model(
                                batch, 
                                energy_func=self.energy_func,
                                ref_model=teacher_model,
                                guidance_iterations=MPGD_N,
                                max_timesteps_for_guidance=max_timesteps_for_guidance,
                                min_guidance_scale=MPGD_W,
                                max_guidance_scale=MPGD_W,
                            )

                            loss = raw_loss / cfg.training.gradient_accumulate_every
                            loss.backward()
                            
                            # step optimizer
                            if (self.global_step % cfg.training.gradient_accumulate_every == 0):
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), cfg.training.max_grad_norm)
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                            
                            # update ema
                            if cfg.training.use_ema:
                                ema.step(accelerator.unwrap_model(self.model))

                            # logging
                            raw_loss_cpu = raw_loss.item()
                            tepoch.set_postfix(
                                loss=raw_loss_cpu,
                                refresh=False
                            )
                            train_losses.append(raw_loss_cpu)
                            step_log = {
                                "train_loss": raw_loss_cpu,
                                "global_step": self.global_step,
                                "epoch": self.epoch,
                                "inner_epoch": inner_epoch,
                            }
                            for k, v in loss_terms.items():
                                step_log[k] = v.item()

                            is_last_batch = batch_idx == (len(dataloader) - 1)
                            if not is_last_batch:
                                # log of last step is combined with validation and rollout
                                accelerator.log(step_log, step=self.global_step)
                                json_logger.log(step_log)
                                self.global_step += 1

                            if (
                                (cfg.training.max_train_steps is not None) 
                                and batch_idx >= (cfg.training.max_train_steps - 1)
                            ):
                                break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = torch.tensor(train_losses, device=device).mean()
                gathered_train_loss = accelerator.gather(train_loss).mean().item()
                step_log["epoch_average_train_loss"] = gathered_train_loss

                # ========= eval for this epoch ==========
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run diffusion sampling on a training batch
                # collected by old model
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = train_sampling_batch
                        obs_dict = batch["obs"]
                        gt_action = batch["action"]

                        result = policy.predict_action(obs_dict)
                        pred_action = result["action_pred"]
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        gathered_mse = accelerator.gather(mse).mean().item()
                        
                        step_log["train_action_mse_error"] = gathered_mse                        
                        
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                accelerator.wait_for_everyone()
                # copy weight to old model
                with torch.no_grad():
                    decay = return_decay(self.global_step, cfg.training.decay_type)
                    step_log["decay"] = decay
                    for param, old_param in zip(
                        self.model.parameters(), self.old_model.parameters(), strict=True):
                        #  decay_type = 0: hard update
                        # old_param.data.copy_(param.detach().clone().data)
                        old_param.data.copy_(
                            old_param.detach().data * decay 
                            + param.detach().clone().data * (1.0 - decay)
                        )

                # checkpoint
                if (
                    ((self.epoch + 1) % cfg.training.checkpoint_every) == 0
                    and accelerator.is_main_process
                ):
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)
                    
                    # checkpointing
                    last_ckpt_path = os.path.join(
                        self.output_dir, "checkpoints", f"{self.epoch + 1}.ckpt"
                    )
                    # Do not save ref_model as it's frozen
                    self.save_checkpoint(last_ckpt_path, exclude_keys=["ref_model"])
                    
                    # retain ddp wrapper
                    self.model = model_ddp

                # end of epoch
                # log of last step is combined with validation and rollout
                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

                try:
                    accelerator.free_memory(dataloader)
                except Exception as e:
                    logger.warning(f"accelerator.free_memory(dataloader) failed: {e}")
                
                try:
                    del dataloader
                    del dataset
                    del train_sampling_batch
                except NameError:
                    pass

                # release VRAM
                gc.collect()
                torch.cuda.empty_cache()

        accelerator.end_training()
        
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
