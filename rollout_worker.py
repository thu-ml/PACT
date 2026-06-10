import sys
import os
import json
import argparse
import shutil
import pickle

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.pkl2hdf5 import (
    parse_dict_structure,
    load_pkl_file,
    append_data_to_structure,
    create_hdf5_from_dict,
)

import h5py
import numpy as np
import yaml
import importlib
from extend.energy_function import init_SM

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


import sys

class DisablePrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance

def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        
        print(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]

def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args

def merge_denosing_pkl_into_hdf5(cache_path, save_dir, ep_num):
    target_file_path = os.path.join(save_dir, f"distillation/episode{ep_num}.hdf5")
    os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
    
    pkl_files = []
    for fname in os.listdir(cache_path):
        if (fname.endswith(".pkl")
            and fname.startswith("denoising_process_")
            and fname.split("_")[2][:-4].isdigit()
        ):
            pkl_files.append((int(fname.split("_")[2][:-4]), os.path.join(cache_path, fname)))
    
    pkl_files.sort()
    pkl_files = [f[1] for f in pkl_files]
    
    # check if the pkl files are consecutive
    expected = 0
    for f in pkl_files:
        num = int(os.path.basename(f).split("_")[2][:-4])
        if num != expected:
            raise ValueError(f"Missing file {expected}.pkl")
        expected += 1
        
    data_list = parse_dict_structure(load_pkl_file(pkl_files[0]))
    for pkl_file_path in pkl_files:
        pkl_file = load_pkl_file(pkl_file_path)
        append_data_to_structure(data_list, pkl_file)

    with h5py.File(target_file_path, "w") as f:
        create_hdf5_from_dict(f, data_list)

def collect_with_policy(task_name,
                        TASK_ENV,
                        args,
                        model,
                        seed_list,
                        collect_num,
                        start_episode_idx,
                        instruction_type,
                        success_only):
    print(f"Task Name: {args['task_name']}")
    print(f"Policy Name: {args['policy_name']}")

    assert args["collect_data"], "collect_data is not set to True"


    # 保存环境信息
    if args["save_env_info"]:
        from extend.save_env_info import get_save_fn_by_task_name
        env_info_save_fn = get_save_fn_by_task_name(args["task_name"])
        print("Environment info will be saved.")
    else:
        env_info_save_fn = None
        print("Environment info will NOT be saved.")

    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = start_episode_idx
    succ_num = 0
    policy_name = args["policy_name"]
    collect_func = eval_function_decorator(policy_name, "collect_data_policy")
    reset_func = eval_function_decorator(policy_name, "reset_model")


    clear_cache_freq = args["clear_cache_freq"]

    print("[Start Collecting Data with Policy]")

    collect_max_steps = args["collect_max_steps"]

    # ensure the training scenes are the same for all methods
    from extend.seed_utils import load_seeds
    seed_path = os.path.join(f"data_task_seeds/{task_name}/demo_randomized/train_seed.txt")
    seed_list_all = load_seeds(seed_path)


    for seed_idx, now_seed in enumerate(seed_list):
        if success_only:
            if TASK_ENV.suc >= collect_num:
                print("Finish Collection")
                break
        else:
            if seed_idx >= collect_num:
                print("Finish Collection")
                break
        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        TASK_ENV.step_lim = collect_max_steps

        seed_idx_in_all = seed_list_all.index(now_seed)

        instructions_file_path = os.path.join(f"data/{task_name}/demo_randomized/instructions/episode{seed_idx_in_all}.json")
        with open(instructions_file_path, "r") as f:
            task_data = json.load(f)

        instruction = np.random.choice(task_data[instruction_type])
        print(instruction)
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        succ = False
        reset_func(model)
        init_SM(TASK_ENV)

        success_frame_index = -1
        more_frames = 10
        with DisablePrint():
            while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
                observation = TASK_ENV.get_obs()
                collect_func(TASK_ENV, model, observation, env_info_save_fn=env_info_save_fn, no_success_break_check=True)

                if TASK_ENV.eval_success and not succ:
                    succ = True
                    success_frame_index = TASK_ENV.take_action_cnt
                    TASK_ENV.eval_success = False

                if succ:
                    TASK_ENV.eval_success = False
                    if TASK_ENV.take_action_cnt >= success_frame_index + more_frames:
                        TASK_ENV.eval_success = True
                        break

        if succ:
            TASK_ENV.suc += 1
            print("Success!")
        else:
            print("Fail!")

        if success_only:
            if succ:
                now_id += 1
        else:
            now_id += 1

        TASK_ENV.close_env(clear_cache=True)
        if success_only:
            if succ:
                TASK_ENV.merge_pkl_to_hdf5_video()
                instruction_save_path = os.path.join(args['save_path'], "instructions", f"episode{TASK_ENV.ep_num}.json")
                os.makedirs(os.path.dirname(instruction_save_path), exist_ok=True)
                shutil.copyfile(instructions_file_path, instruction_save_path)

        else:
            if args.get('stagecut', False) and 'stage' in observation['env']:
                if not succ:
                    last_stage = observation['env']['stage']

                    cache_file_path = TASK_ENV.folder_path["cache"]
                    cache_file_list = os.listdir(cache_file_path)
                    cache_file_list = [fname for fname in cache_file_list if fname.endswith(".pkl")]
                    cache_file_list_sorted = sorted(cache_file_list, key=lambda x: int(x.split(".")[0]))

                    keep_num = 50
                    first_same_stage_idx = len(cache_file_list_sorted)
                    for idx, fname in enumerate(cache_file_list_sorted):
                        with open(os.path.join(cache_file_path, fname), "rb") as f:
                            pkl_data = pickle.load(f)
                        assert 'stage' in pkl_data['env'], "No stage info in pkl file"
                        if pkl_data['env']['stage'] == last_stage:
                            first_same_stage_idx = idx
                            break


                    for fname in cache_file_list_sorted:
                        with open(os.path.join(cache_file_path, fname), "rb") as f:
                            pkl_data = pickle.load(f)
                        assert 'stage' in pkl_data['env'], "No stage info in pkl file"

                        idx_of_file = int(fname.split(".")[0])
                        if pkl_data['env']['stage'] >= last_stage and idx_of_file >= first_same_stage_idx + keep_num:
                            os.remove(os.path.join(cache_file_path, fname))
                            print(f"Removed pkl file: {fname} for failed episode {TASK_ENV.ep_num} at stage {last_stage}")



            TASK_ENV.merge_pkl_to_hdf5_video()
            instruction_save_path = os.path.join(args['save_path'], "instructions", f"episode{TASK_ENV.ep_num}.json")
            os.makedirs(os.path.dirname(instruction_save_path), exist_ok=True)
            shutil.copyfile(instructions_file_path, instruction_save_path)
            
        TASK_ENV.remove_data_cache_by_episode_idx(TASK_ENV.ep_num)

        TASK_ENV.test_num += 1

        print(
            f"{task_name} | {args['policy_name']} | {args['task_config']} | {args['ckpt_setting']}\n"
            f"Success rate: {TASK_ENV.suc}/{TASK_ENV.test_num} => {round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%, current seed: {now_seed}\n"
        )

def worker(usr_args):
    pid = os.getpid()
    print(f"Process PID: {pid}")
    task_idx = usr_args["task_idx"]
    seed_list = usr_args["seed_list"]
    np.random.shuffle(seed_list)
    ckpt_path = usr_args["ckpt_path"]
    collect_max_steps = usr_args["collect_max_steps"]
    print(f"process {task_idx} start")
    print(f"process {task_idx} seeds:", seed_list)
    print(f"ckpt path: {ckpt_path}")
    print(usr_args)

    from script.test_render import Sapien_TEST
    Sapien_TEST()

    
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    collect_num = usr_args["collect_num"]
    success_only = usr_args["success_only"]
    start_episode_idx = task_idx * collect_num

    get_model_by_path = eval_function_decorator(policy_name, "get_model_by_path")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    
    # output camera config
    print("============= Config =============\n")
    print("Messy Table: " + str(args["domain_randomization"]["cluttered_table"]))
    print("Random Background: " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("Random Light: " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("Random Table Height: " + str(args["domain_randomization"]["random_table_height"]))
    print("Random Head Camera Distance: " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("Head Camera Config: " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("Wrist Camera Config: " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("Embodiment Config: " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["save_path"] = usr_args["save_path"]
    args["save_data"] = True
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    args['save_env_info'] = usr_args.get('save_env_info', False)

    model = get_model_by_path(
        ckpt_path,
        usr_args)
    
    args["collect_max_steps"] = collect_max_steps

    args['stagecut'] = usr_args.get('stagecut', False)

    if  args['stagecut']:
        print("Stage Cut is Enabled for Failed Episode Handling.")
    else:
        print("Stage Cut is Disabled.")

    collect_with_policy(task_name,
                        TASK_ENV,
                        args,
                        model,
                        seed_list,
                        collect_num,
                        start_episode_idx,
                        instruction_type,
                        success_only)
    with open(os.path.join(usr_args["save_path"], 'logs', f'signal_{task_idx}'), 'w') as f:
        f.write(f"This is a finished signal for task {task_idx}.")
    print(f"process {task_idx} finish")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--args_path", type=str, required=True)
    parser.add_argument("--task_idx", type=str, required=True)
    parser.add_argument("--success_only", action="store_true")
    args = parser.parse_args()
    arg_file_path = os.path.join(args.args_path, f'{args.task_idx}.json')
    with open(arg_file_path, "r") as f:
        usr_args = json.load(f)
    usr_args['success_only'] = args.success_only
    worker(usr_args)