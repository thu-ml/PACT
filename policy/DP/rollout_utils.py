import os
import shutil
from collections import defaultdict

import cv2
import h5py
import numpy as np
import zarr
import re
import gc

def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        vector = root["/joint_action/vector"][()]
        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]
        

        env = root["/env"]
        env_dict = dict()
        all_keys = list(env.keys())
        for k in all_keys:
            env_dict[k] = env[k][()]

    return left_gripper, left_arm, right_gripper, right_arm, vector, image_dict, env_dict


def process_data(load_dir, save_dir):
    num = len(os.listdir(os.path.join(load_dir, "data")))
    total_count = 0

    if os.path.exists(save_dir):
        print(f"Save dir {save_dir} already exists! Remove it first.")
        shutil.rmtree(save_dir)

    current_ep = 0

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    head_camera_arrays, front_camera_arrays, left_camera_arrays, right_camera_arrays = (
        [],
        [],
        [],
        [],
    )
    episode_ends_arrays, action_arrays, state_arrays, joint_action_arrays = (
        [],
        [],
        [],
        [],
    )


    env_arrays = defaultdict(list)
    while current_ep < num:
        print(f"processing episode: {current_ep + 1} / {num}", end="\r")

        load_path = os.path.join(load_dir, f"data/episode{current_ep}.hdf5")
        (
            left_gripper_all,
            left_arm_all,
            right_gripper_all,
            right_arm_all,
            vector_all,
            image_dict_all,
            env_dict_all,
        ) = load_hdf5(load_path)

        for j in range(0, left_gripper_all.shape[0]):

            head_img_bit = image_dict_all["head_camera"][j]
            joint_state = vector_all[j]

            env_dict_now = dict()
            for k, v in env_dict_all.items():
                env_dict_now[k] = v[j]

            if j != left_gripper_all.shape[0] - 1:
                head_img = cv2.imdecode(np.frombuffer(head_img_bit, np.uint8), cv2.IMREAD_COLOR)
                head_camera_arrays.append(head_img)
                state_arrays.append(joint_state)
                for key in env_dict_now.keys():
                    env_arrays[key].append(env_dict_now[key])
                
            if j != 0:
                joint_action_arrays.append(joint_state)
        current_ep += 1
        total_count += left_gripper_all.shape[0] - 1
        episode_ends_arrays.append(total_count)

    episode_ends_arrays = np.array(episode_ends_arrays)
    state_arrays = np.array(state_arrays)
    head_camera_arrays = np.array(head_camera_arrays)
    joint_action_arrays = np.array(joint_action_arrays)

    for key in env_arrays.keys():
        env_arrays[key] = np.array(env_arrays[key], dtype=np.float32)

    head_camera_arrays = np.moveaxis(head_camera_arrays, -1, 1)  # NHWC -> NCHW

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    # action_chunk_size = (100, action_arrays.shape[1])
    state_chunk_size = (100, state_arrays.shape[1])
    joint_chunk_size = (100, joint_action_arrays.shape[1])
    head_camera_chunk_size = (100, *head_camera_arrays.shape[1:])
    zarr_data.create_dataset(
        "head_camera",
        data=head_camera_arrays,
        chunks=head_camera_chunk_size,
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "state",
        data=state_arrays,
        chunks=state_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "action",
        data=joint_action_arrays,
        chunks=joint_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    if len(env_arrays) > 0:
        for key in env_arrays.keys():
            assert key not in zarr_data.keys(), f"Key '{key}' already exists in zarr_data"
            zarr_data.create_dataset(
                key,
                data=env_arrays[key],
                chunks=(100, *env_arrays[key].shape[1:]),
                dtype="float32",
                overwrite=True,
                compressor=compressor,
            )

    zarr_meta.create_dataset(
        "episode_ends",
        data=episode_ends_arrays,
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )

    del head_camera_arrays
    del state_arrays
    del joint_action_arrays
    del env_arrays
    del episode_ends_arrays

    try:
        del left_gripper_all, left_arm_all, right_gripper_all, right_arm_all
        del vector_all, image_dict_all, env_dict_all
    except NameError:
        pass

    gc.collect()


def compute_success_rate(save_path):
    stdouts_path = os.path.join(save_path, 'stdouts')
    out_files = os.listdir(stdouts_path)
    
    out_files.sort(key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else -1)
    
    success_seeds = []
    fail_seeds = []
    
    for file_name in out_files:
        file_path = os.path.join(stdouts_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            success_rate_matches = re.findall(r'Success rate: (\d+)/(\d+) => [\d.]+%, current seed: (\d+)', content)
            previous_success_count = 0
            for match in success_rate_matches:
                success_count = int(match[0])
                total_count = int(match[1])
                seed = int(match[2])
                if previous_success_count < success_count:
                    success_seeds.append(seed)
                else:
                    fail_seeds.append(seed)
                previous_success_count = success_count

    total_success = len(success_seeds)
    total_fail = len(fail_seeds)
    total_tasks = total_success + total_fail
    
    assert total_tasks > 0
    success_rate = (total_success / total_tasks)
    with open(os.path.join(save_path, "logs", "rollout_result.txt"), "w") as f:
        f.writelines([
            f"Find {len(out_files)} .out files: {out_files}\n",
            "Results:\n",
            f"Success Seeds ({len(success_seeds)}个):\n",
            '[' + ','.join(map(str, sorted(success_seeds))) + ']\n',
            f"Fail Seeds ({len(fail_seeds)}个):\n",
            '[' + ','.join(map(str, sorted(fail_seeds))) + ']\n',
            f"Total Task Number: {total_tasks}\n",
            f"Success Number: {total_success}\n",
            f"Fail Number: {total_fail}\n",
            f"Success Rate: {(success_rate * 100):.2f}%\n"])
    return success_rate