import numpy as np
from .dp_model import DP
import yaml
import os
from envs.utils import *

def encode_obs(observation):
    head_cam = (np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0) / 255)
    left_cam = (np.moveaxis(observation["observation"]["left_camera"]["rgb"], -1, 0) / 255)
    right_cam = (np.moveaxis(observation["observation"]["right_camera"]["rgb"], -1, 0) / 255)
    obs = dict(
        head_cam=head_cam,
        left_cam=left_cam,
        right_cam=right_cam,
    )
    obs["agent_pos"] = observation["joint_action"]["vector"]
    return obs


def get_model(usr_args):
    action_dim = usr_args['left_arm_dim'] + usr_args['right_arm_dim'] + 2 # 2 gripper
    
    if usr_args.get("ckpt_file", None) is not None:
        print("[WARNING] ckpt_file is provided, will override the ckpt_file from the config file")
        ckpt_file = usr_args["ckpt_file"]  
    elif usr_args.get("eval_distilled_model", False):  
        assert usr_args["exp_name"] is not None, "exp_name is required"
        print(f"Evaluating distilled model with exp_name: ")
        ckpt_file = f"playground/dp/{usr_args['task_name']}/{usr_args['exp_name']}_robot_on_policy_distillation_{usr_args['task_name']}/checkpoints/{usr_args['checkpoint_num']}.ckpt"
    else:
        ckpt_file = f"./policy/DP/checkpoints/{usr_args['task_name']}-{usr_args['ckpt_setting']}-{usr_args['expert_data_num']}-{usr_args['seed']}/{usr_args['checkpoint_num']}.ckpt"
    
    load_config_path = f'./policy/DP/diffusion_policy/config/robot_dp_{action_dim}.yaml'
    with open(load_config_path, "r", encoding="utf-8") as f:
        model_training_config = yaml.safe_load(f)
    
    n_obs_steps = model_training_config['n_obs_steps']
    n_action_steps = model_training_config['n_action_steps']
    print(f'load ckpt from {ckpt_file}')
    return DP(ckpt_file, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)

def get_model_by_path(ckpt_file, usr_args):
    action_dim = usr_args['left_arm_dim'] + usr_args['right_arm_dim'] + 2 # 2 gripper
    load_config_path = f'./policy/DP/diffusion_policy/config/robot_dp_{action_dim}.yaml'
    with open(load_config_path, "r", encoding="utf-8") as f:
        model_training_config = yaml.safe_load(f)
    
    n_obs_steps = model_training_config['n_obs_steps']
    n_action_steps = model_training_config['n_action_steps']
    print(f'load ckpt from {ckpt_file} with use_ema: {usr_args.get("use_ema", None)}')
    return DP(ckpt_file, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps, use_ema=usr_args.get("use_ema", None))


def eval(TASK_ENV, model, observation, test_danger=False):
    """
    TASK_ENV: Task Environment Class, you can use this class to interact with the environment
    model: The model from 'get_model()' function
    observation: The observation about the environment
    """
    obs = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()

    # ======== Get Action ========
    actions = model.get_action(obs)

    for idx, action in enumerate(actions):
        TASK_ENV.take_action(action, test_danger=test_danger)
        observation = TASK_ENV.get_obs()
        TASK_ENV.FRAME_IDX += 1
        obs = encode_obs(observation)
        if idx < len(actions)-1:
            model.update_obs(obs)

def collect_data_policy(
    TASK_ENV, model, observation, 
    env_info_save_fn=None,
    no_success_break_check=False,
):
    """
    TASK_ENV: Task Environment Class, you can use this class to interact with the environment
    model: The model from 'get_model()' function
    observation: The observation about the environment
    """
    obs = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()

    # ======== Get Action with Guidance========
    actions = model.get_action(obs)
   
    if TASK_ENV.FRAME_IDX == 0:
        TASK_ENV.folder_path = {"cache": f"{TASK_ENV.save_dir}/.cache/episode{TASK_ENV.ep_num}/"}

        for directory in TASK_ENV.folder_path.values():  # remove previous data
            if os.path.exists(directory):
                file_list = os.listdir(directory)
                for file in file_list:
                    os.remove(directory + file)
    
    eval_successed = False
    for idx, action in enumerate(actions):
        # print(TASK_ENV.save_dir, TASK_ENV.ep_num, TASK_ENV.FRAME_IDX)
        if env_info_save_fn is not None:
            # save required env info to calculate energy guidance
            # env_info = env_info_save_fn(TASK_ENV, return_tensors="np")
            env_info = env_info_save_fn(TASK_ENV, return_tensors="np", action=action, action_chunk=actions)
            observation['env'] = env_info
        save_pkl(TASK_ENV.folder_path["cache"] + f"{TASK_ENV.FRAME_IDX}.pkl", observation)  # use cache

        TASK_ENV.take_action(action, no_success_break_check=no_success_break_check) 

        observation = TASK_ENV.get_obs()
        TASK_ENV.FRAME_IDX += 1
        obs = encode_obs(observation)
        if idx < len(actions) - 1:
            model.update_obs(obs)
        
        if TASK_ENV.check_success():
            eval_successed = True
    
    if eval_successed:
        print(f"Episode {TASK_ENV.ep_num} successed!")
        TASK_ENV.eval_success = True

def reset_model(model):
    model.reset_obs()
