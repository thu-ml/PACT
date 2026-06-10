import torch
from typing import Dict, Any
from extend.forward_kinematics import get_gripper_keypoints
from extend.safe_constraint import (build_cube_side_normals, quaternion_to_direction)
import transforms3d as t3d
import numpy as np

def convert_to_tensors(tensor_dict, return_tensors="pt"):
    """
    convert tensor dict to tensors or numpy array
    Assume all inputs are tensors in pytorch or floats.
    
    Args:
        tensor_dict: dictionary of tensors or numpy arrays
        return_tensors: "pt" or "np", return tensors in pytorch or numpy
        
    Returns:
        tensor_dict: dictionary of tensors or numpy arrays
    """
    if return_tensors != "np":
        return tensor_dict
    
    # return_tensors == "np"
    for key in tensor_dict:
        if isinstance(tensor_dict[key], torch.Tensor):
            tensor_dict[key] = tensor_dict[key].numpy()
    
    return tensor_dict
    
def add_base_info(TASK_ENV, env_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add base info to the environment info.
    
    Args:
        TASK_ENV: Task Environment Class
        env_info: Dictionary of environment info
        
    Returns:
        env_info: Dictionary of environment info
    """
    # left & right arm
    env_info['S_left'] = TASK_ENV.S_left.clone().cpu()
    env_info['M_left'] = TASK_ENV.M_left.clone().cpu()
    env_info['S_right'] = TASK_ENV.S_right.clone().cpu()
    env_info['M_right'] = TASK_ENV.M_right.clone().cpu()
    
    return env_info

def add_gripper_bias(TASK_ENV, env_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add gripper bias to the environment info.
    
    Args:
        TASK_ENV: Task Environment Class
        env_info: Dictionary of environment info
        
    Returns:
        env_info: Dictionary of environment info
    """
    env_info['left_gripper_bias'] = TASK_ENV.robot.left_gripper_bias    # float
    env_info['right_gripper_bias'] = TASK_ENV.robot.right_gripper_bias    # float
    return env_info

def get_meta_info(TASK_ENV, return_tensors="pt"):
    """
    Get meta info of the task environment.
    Meta info is time-independent params, can be saved as meta info.
    
    Args:
        TASK_ENV: Task Environment Class
        return_tensors: "pt" or "np", return tensors in pytorch or numpy
    """
    meta_info = {}
    
    # add base info
    meta_info = add_base_info(TASK_ENV, meta_info)
    # add gripper bias
    meta_info = add_gripper_bias(TASK_ENV, meta_info)
    
    # convert to tensors or numpy array
    meta_info = convert_to_tensors(meta_info, return_tensors)
    
    return meta_info

def get_save_fn_by_task_name(task_name):
    """
    Get save env info function by task name.
    
    Args:
        task_name: Name of the task
        
    Returns:
        save_env_info_fn: Function to save env info
    """
    if task_name == 'pick_dual_bottles' or task_name == 'pick_diverse_bottles':
        return save_env_info_for_pick_dual_bottles
    elif task_name == 'handover_apple':
        return save_env_info_for_handover_apple
    elif task_name == 'handover_block':
        return save_env_info_for_handover_block
    elif task_name == 'stack_blocks_two':
        return save_env_info_for_stack_blocks_two
    elif task_name == 'pour_water_to_cup':
        return save_env_info_for_pour_water_to_cup
    elif task_name == 'place_dual_shoes':
        return save_env_info_for_place_dual_shoes
    else:
        raise ValueError(f"Unsupported save env info task name: {task_name}")

class GuidanceStateRecorder:
    def __init__(self):
        pass

def save_env_info_for_pick_dual_bottles(
    TASK_ENV, return_tensors, action=None, action_chunk=None, device='cuda'
):
    """
    Save env info for pick dual bottles task.
    
    Args:
        TASK_ENV: task environment class
        return_tensors: "pt" or "np", return tensors in pytorch or numpy
    """
    env_info = {}
    
    # define task-specific info here
    # bottle1 & bottle2    
    env_info['bottle1_functional_point'] = torch.tensor(TASK_ENV.bottle1.get_functional_point(0))
    env_info['bottle2_functional_point'] = torch.tensor(TASK_ENV.bottle2.get_functional_point(0))
    
    # convert to tensors or numpy array
    env_info = convert_to_tensors(env_info, return_tensors)
    
    return env_info

def save_env_info_for_handover_apple(
    TASK_ENV, return_tensors, action, action_chunk, device='cuda'
):
    """
    Save env info for handover apple task.
    
    Args:
        TASK_ENV: Task Environment Class
        return_tensors: "pt" or "np", return tensors in pytorch or numpy
    """
    env_info = {}
    assert action_chunk is not None, "Action chunk must be provided for handover apple task."

    trajectory_unnormed = action_chunk

    if type(trajectory_unnormed) is not torch.Tensor:
        trajectory_unnormed = torch.tensor(trajectory_unnormed, device=device)
    
    S_left = TASK_ENV.S_left.to(device)
    M_left = TASK_ENV.M_left.to(device)
    S_right = TASK_ENV.S_right.to(device)
    M_right = TASK_ENV.M_right.to(device)


    apple_pos = torch.tensor(TASK_ENV.apple.get_contact_point(0), device=device)[:3]

    actions = trajectory_unnormed

    left_gripper_positions, left_gripper_directions = get_gripper_keypoints(S_left, M_left, actions[:,:7], TASK_ENV.robot.left_gripper_bias)
    right_gripper_positions, right_gripper_directions = get_gripper_keypoints(S_right, M_right, actions[:,7:], TASK_ENV.robot.right_gripper_bias)

    if TASK_ENV.FRAME_IDX == 0:
        TASK_ENV.guidance_state_recorder = GuidanceStateRecorder()
        TASK_ENV.guidance_state_recorder.l_poking = False
        TASK_ENV.guidance_state_recorder.r_poking = False
        TASK_ENV.guidance_state_recorder.apple_pos_org = apple_pos
        TASK_ENV.guidance_state_recorder.stage = 0


    if TASK_ENV.guidance_state_recorder.stage == 0:
        if torch.norm(left_gripper_positions["forward"][0]-apple_pos) < 0.2:
            TASK_ENV.guidance_state_recorder.l_poking = True
            TASK_ENV.guidance_state_recorder.stage = 1
    if TASK_ENV.guidance_state_recorder.stage == 1:
        if apple_pos[2] - TASK_ENV.guidance_state_recorder.apple_pos_org[2] > 0.04:
            TASK_ENV.guidance_state_recorder.l_poking = False
            TASK_ENV.guidance_state_recorder.stage = 2
    if TASK_ENV.guidance_state_recorder.stage == 2:
        if torch.norm(right_gripper_positions["forward"][0]-apple_pos) < 0.2:
            TASK_ENV.guidance_state_recorder.r_poking = True
            TASK_ENV.guidance_state_recorder.stage = 3

    env_info = {
        'l_poking': TASK_ENV.guidance_state_recorder.l_poking,
        'r_poking': TASK_ENV.guidance_state_recorder.r_poking,
        'apple_pos': apple_pos.to('cpu'),
        'stage': TASK_ENV.guidance_state_recorder.stage,
    }

    convert_to_tensors(env_info, return_tensors)

    return env_info

def save_env_info_for_handover_block(
    TASK_ENV, return_tensors, action, action_chunk, device='cuda'
):
    """
    Save env info for handover block task.
    
    Args:
        TASK_ENV: task environment class
        return_tensors: "pt" or "np", return tensors in pytorch or numpy
    """
    env_info = {}
    assert action_chunk is not None, "Action chunk must be provided for handover block task."

    trajectory_unnormed = action_chunk

    if type(trajectory_unnormed) is not torch.Tensor:
        trajectory_unnormed = torch.tensor(trajectory_unnormed, device=device)
    
    S_left = TASK_ENV.S_left.to(device)
    M_left = TASK_ENV.M_left.to(device)
    S_right = TASK_ENV.S_right.to(device)
    M_right = TASK_ENV.M_right.to(device)

    box_pos_top = torch.tensor(TASK_ENV.box.get_contact_point(0), device=device)[:3]
    box_pos_bot = torch.tensor(TASK_ENV.box.get_contact_point(5), device=device)[:3]

    actions = trajectory_unnormed

    left_gripper_positions, left_gripper_directions = get_gripper_keypoints(S_left, M_left, actions[:,:7], TASK_ENV.robot.left_gripper_bias)
    right_gripper_positions, right_gripper_directions = get_gripper_keypoints(S_right, M_right, actions[:,7:], TASK_ENV.robot.right_gripper_bias)

    if TASK_ENV.FRAME_IDX == 0:
        TASK_ENV.guidance_state_recorder = GuidanceStateRecorder()
        TASK_ENV.guidance_state_recorder.l_poking = True
        TASK_ENV.guidance_state_recorder.r_distance = False
        TASK_ENV.guidance_state_recorder.box_pos_top_org = box_pos_top
        TASK_ENV.guidance_state_recorder.stage = 0
    
    if TASK_ENV.guidance_state_recorder.stage == 0:
        if box_pos_top[2] - TASK_ENV.guidance_state_recorder.box_pos_top_org[2] > 0.05:
            TASK_ENV.guidance_state_recorder.l_poking = False
            TASK_ENV.guidance_state_recorder.stage = 1

    if TASK_ENV.guidance_state_recorder.stage == 1:
        if actions[:,:7].std(dim=0).max()<0.01:
            TASK_ENV.guidance_state_recorder.r_distance = True
            TASK_ENV.guidance_state_recorder.stage = 2

    if TASK_ENV.guidance_state_recorder.stage == 2: 
        if right_gripper_positions["forward"][0,0] < 0.29:
            TASK_ENV.guidance_state_recorder.r_distance = False
            TASK_ENV.guidance_state_recorder.stage = 3

    env_info = {
        'l_poking': TASK_ENV.guidance_state_recorder.l_poking,
        'r_distance': TASK_ENV.guidance_state_recorder.r_distance,
        'box_pos_top': box_pos_top.to('cpu'),
        'box_pos_bot': box_pos_bot.to('cpu'),
    }

    convert_to_tensors(env_info, return_tensors)

    return env_info

def save_env_info_for_stack_blocks_two(
    TASK_ENV, return_tensors, action, action_chunk, device='cuda'
):
    """
    Save env info for stack blocks two task.
    
    Args:
        TASK_ENV: task environment class
        return_tensors: "pt" or "np", return tensors in pytorch or numpy
    """
    env_info = {}
    assert action_chunk is not None, "Action chunk must be provided for handover block task."

    trajectory_unnormed = action_chunk

    if type(trajectory_unnormed) is not torch.Tensor:
        trajectory_unnormed = torch.tensor(trajectory_unnormed, device=device)

    S_left = TASK_ENV.S_left.to(device)
    M_left = TASK_ENV.M_left.to(device)
    S_right = TASK_ENV.S_right.to(device)
    M_right = TASK_ENV.M_right.to(device)

    block1_pos = torch.tensor(TASK_ENV.block1.get_contact_point(0), device=device, dtype=torch.float64)[:3]
    block1_dir = torch.tensor(TASK_ENV.block1.get_contact_point(0), device=device, dtype=torch.float64)[3:]
    block1_dir = quaternion_to_direction(block1_dir)
    block1_dirs = build_cube_side_normals(block1_dir)
    block2_pos = torch.tensor(TASK_ENV.block2.get_contact_point(0), device=device, dtype=torch.float64)[:3]
    block2_dir = torch.tensor(TASK_ENV.block2.get_contact_point(0), device=device, dtype=torch.float64)[3:]
    block2_dir = quaternion_to_direction(block2_dir)
    block2_dirs = build_cube_side_normals(block2_dir)

    actions = trajectory_unnormed

    if TASK_ENV.FRAME_IDX == 0:
        TASK_ENV.guidance_state_recorder = GuidanceStateRecorder()
        TASK_ENV.guidance_state_recorder.b1_poking = True
        TASK_ENV.guidance_state_recorder.b2_poking = False
        TASK_ENV.guidance_state_recorder.b1_alignment = False
        TASK_ENV.guidance_state_recorder.b2_alignment = False
        TASK_ENV.guidance_state_recorder.stage = 0
        TASK_ENV.guidance_state_recorder.block1_pos_org = block1_pos
        TASK_ENV.guidance_state_recorder.block2_pos_org = block2_pos
        TASK_ENV.guidance_state_recorder.b1_left = True if block1_pos[0] < 0 else False
        TASK_ENV.guidance_state_recorder.b2_left = True if block2_pos[0] < 0 else False

    if TASK_ENV.guidance_state_recorder.b1_left:
        block1_gripper_positions, block1_gripper_directions = get_gripper_keypoints(S_left, M_left, actions[:,:7], TASK_ENV.robot.left_gripper_bias)
    else:
        block1_gripper_positions, block1_gripper_directions = get_gripper_keypoints(S_right, M_right, actions[:,7:], TASK_ENV.robot.right_gripper_bias)

    if TASK_ENV.guidance_state_recorder.stage == 0:
        if block1_pos[2] - TASK_ENV.guidance_state_recorder.block1_pos_org[2] > 0.04:
            TASK_ENV.guidance_state_recorder.b1_poking = False
            TASK_ENV.guidance_state_recorder.b1_alignment = True
            TASK_ENV.guidance_state_recorder.stage = 1

    if TASK_ENV.guidance_state_recorder.stage == 1:
        if block1_pos[2] - TASK_ENV.guidance_state_recorder.block1_pos_org[2] < 0.005:
            TASK_ENV.guidance_state_recorder.b1_alignment = False
            TASK_ENV.guidance_state_recorder.stage = 2
            
    if TASK_ENV.guidance_state_recorder.stage == 2:
        if torch.norm(block1_gripper_positions["forward"][0]-block1_pos) > 0.1:
            TASK_ENV.guidance_state_recorder.b2_poking = True
            TASK_ENV.guidance_state_recorder.stage = 3
            

    if TASK_ENV.guidance_state_recorder.stage == 3:
        if block2_pos[2] - TASK_ENV.guidance_state_recorder.block2_pos_org[2] > 0.04:
            TASK_ENV.guidance_state_recorder.b2_poking = False
            TASK_ENV.guidance_state_recorder.b2_alignment = True
            TASK_ENV.guidance_state_recorder.stage = 4

    env_info = {
        'block1_pos': block1_pos.to('cpu'),
        'block2_pos': block2_pos.to('cpu'),
        'block1_dirs': block1_dirs.to('cpu'),
        'block2_dirs': block2_dirs.to('cpu'),
        'b1_left': TASK_ENV.guidance_state_recorder.b1_left,
        'b2_left': TASK_ENV.guidance_state_recorder.b2_left,
        'b1_poking': TASK_ENV.guidance_state_recorder.b1_poking,
        'b2_poking': TASK_ENV.guidance_state_recorder.b2_poking,
        'b1_alignment': TASK_ENV.guidance_state_recorder.b1_alignment,
        'b2_alignment': TASK_ENV.guidance_state_recorder.b2_alignment,
        'table_z_bias': TASK_ENV.table_z_bias,
    }

    convert_to_tensors(env_info, return_tensors)

    return env_info

def save_env_info_for_pour_water_to_cup(
    TASK_ENV, return_tensors, action, action_chunk, device='cuda'
):
    """
    Save env info for pour water to cup task.
    
    Args:
        TASK_ENV: Task Environment Class
        return_tensors: "pt" or "np", return tensors in pytorch or numpy
    """

    trajectory_unnormed = action_chunk[None]
    
    if type(trajectory_unnormed) is not torch.Tensor:
        trajectory_unnormed = torch.tensor(trajectory_unnormed, device=device)

    S_left = TASK_ENV.S_left.to(trajectory_unnormed.device)
    M_left = TASK_ENV.M_left.to(trajectory_unnormed.device)
    S_right = TASK_ENV.S_right.to(trajectory_unnormed.device)
    M_right = TASK_ENV.M_right.to(trajectory_unnormed.device)
    left_arm_dim = S_left.shape[-1]
    right_arm_dim = S_right.shape[-1]

    bottle_pos = torch.tensor(TASK_ENV.bottle.get_functional_point(0), device=trajectory_unnormed.device)[:3]
    cup_pos = torch.tensor(TASK_ENV.cup.get_contact_point(0), device=trajectory_unnormed.device)[:3]

    q_gripper_left = trajectory_unnormed[0, :, left_arm_dim]
    q_gripper_right = trajectory_unnormed[0, :, left_arm_dim + right_arm_dim + 1]
   
    smooth_l1_loss = torch.nn.SmoothL1Loss(reduction='none', beta=0.1)
    distance_process = lambda x: smooth_l1_loss(x, torch.zeros_like(x))

    actions = trajectory_unnormed[0]
    with torch.no_grad():
        left_gripper_positions, left_gripper_directions = get_gripper_keypoints(S_left, M_left, actions[:,:7], TASK_ENV.robot.left_gripper_bias)
        right_gripper_positions, right_gripper_directions = get_gripper_keypoints(S_right, M_right, actions[:,7:], TASK_ENV.robot.right_gripper_bias)
        
    def object_in_hand(object_pos, gripper_positions, threshold=0.03):
        time_step = 0
        return torch.norm(object_pos - gripper_positions["forward"][time_step], p=2) < threshold

    if TASK_ENV.FRAME_IDX == 0:
        TASK_ENV.guidance_state_recorder = GuidanceStateRecorder()
        TASK_ENV.guidance_state_recorder.l_poking = True
        TASK_ENV.guidance_state_recorder.r_poking = True
        TASK_ENV.guidance_state_recorder.l_tilt = False
        TASK_ENV.guidance_state_recorder.cup_init_pos = cup_pos.clone().detach()
        TASK_ENV.guidance_state_recorder.bottle_init_pos = bottle_pos.clone().detach()
        TASK_ENV.guidance_state_recorder.stage = 0
    
    if TASK_ENV.guidance_state_recorder.stage == 0:
        if q_gripper_left[0] < 0.1 and q_gripper_right[0] < 0.1:
            bottle_in_hand = object_in_hand(bottle_pos, left_gripper_positions, threshold=0.03)
            cup_in_hand = object_in_hand(cup_pos, right_gripper_positions, threshold=0.03)
            bottle_lifted = torch.norm(bottle_pos - TASK_ENV.guidance_state_recorder.bottle_init_pos, p=2, dim=0) > 0.05
            cup_lifted = torch.norm(cup_pos - TASK_ENV.guidance_state_recorder.cup_init_pos, p=2, dim=0) > 0.05
            if bottle_in_hand and cup_in_hand and bottle_lifted and cup_lifted:
                TASK_ENV.guidance_state_recorder.stage = 1
                TASK_ENV.guidance_state_recorder.l_poking = False
                TASK_ENV.guidance_state_recorder.r_poking = False

    if TASK_ENV.guidance_state_recorder.stage == 1:
        if q_gripper_left[0] < 0.1 and q_gripper_right[0] < 0.1:
            bottle_in_hand = object_in_hand(bottle_pos, left_gripper_positions, threshold=0.03)
            cup_in_hand = object_in_hand(cup_pos, right_gripper_positions, threshold=0.03)
            
            left_target_position = [-0.1, 0, TASK_ENV.guidance_state_recorder.bottle_init_pos[2]+0.1]
            right_target_position = [0.1, 0, TASK_ENV.guidance_state_recorder.cup_init_pos[2]+0.1]
            bottle_to_target_dist = torch.norm(bottle_pos - torch.tensor(left_target_position, device=trajectory_unnormed.device), p=2, dim=0)
            cup_to_target_dist = torch.norm(cup_pos - torch.tensor(right_target_position, device=trajectory_unnormed.device), p=2, dim=0)

            future_pos_left = left_gripper_positions["forward"]
            left_pos_movement = torch.norm(future_pos_left[-1] - future_pos_left[0], p=2, dim=0)
            movement_threshold = 0.05
            left_hand_stable = left_pos_movement < movement_threshold

            if bottle_in_hand and cup_in_hand and bottle_to_target_dist < 0.05 and cup_to_target_dist < 0.05 and left_hand_stable:
                TASK_ENV.guidance_state_recorder.stage = 2
                TASK_ENV.guidance_state_recorder.l_tilt = True
                
    bottle_top_to_functional_offset = ((torch.tensor(TASK_ENV.bottle.config['extents'], device=trajectory_unnormed.device)-torch.tensor(TASK_ENV.bottle.config['center'], device=trajectory_unnormed.device)) * torch.tensor(TASK_ENV.bottle.config['scale'], device=trajectory_unnormed.device))[1]
    bottle_up_direction = left_gripper_directions['top']
    bottle_up_offset = bottle_up_direction / torch.norm(bottle_up_direction, dim=1, keepdim=True) * bottle_top_to_functional_offset
    bottle_top_positions = left_gripper_positions['forward'] + bottle_up_offset

    cup_top_to_functional_offset = ((torch.tensor(TASK_ENV.cup.config['extents'], device=trajectory_unnormed.device)-torch.tensor(TASK_ENV.cup.config['center'], device=trajectory_unnormed.device)) * torch.tensor(TASK_ENV.cup.config['scale'], device=trajectory_unnormed.device))[1]
    cup_up_direction = right_gripper_directions['top']
    cup_up_offset = cup_up_direction / torch.norm(cup_up_direction, dim=1, keepdim=True) * cup_top_to_functional_offset
    cup_top_positions = right_gripper_positions['forward'] + cup_up_offset

    if TASK_ENV.guidance_state_recorder.stage == 2:
        bottle_top_to_cup_top_dist = torch.norm(bottle_top_positions - cup_top_positions, p=2, dim=1)

        if (bottle_top_to_cup_top_dist < 0.05).sum()>1:
            TASK_ENV.guidance_state_recorder.stage = 4
            TASK_ENV.done = True

    env_info = {
        'l_poking': TASK_ENV.guidance_state_recorder.l_poking,
        'r_poking': TASK_ENV.guidance_state_recorder.r_poking,
        'l_tilt': TASK_ENV.guidance_state_recorder.l_tilt,
        'bottle_pos': bottle_pos.to('cpu'),
        'cup_pos': cup_pos.to('cpu'),
        'bottle_top_to_functional_offset': bottle_top_to_functional_offset.to('cpu'),
        'cup_top_to_functional_offset': cup_top_to_functional_offset.to('cpu'),
        'stage': TASK_ENV.guidance_state_recorder.stage,
    }
    convert_to_tensors(env_info, return_tensors)
    
    return env_info

def save_env_info_for_place_dual_shoes(
    TASK_ENV, return_tensors, action, action_chunk, device='cuda'
):
    """
    Save env info for place dual shoes task.
    
    Args:
        TASK_ENV: Task Environment Class
        return_tensors: "pt" or "np", return tensors in pytorch or numpy
    """

    trajectory_unnormed = action_chunk[None]
    
    if type(trajectory_unnormed) is not torch.Tensor:
        trajectory_unnormed = torch.tensor(trajectory_unnormed, device=device)

    S_left = TASK_ENV.S_left.to(trajectory_unnormed.device)
    M_left = TASK_ENV.M_left.to(trajectory_unnormed.device)
    S_right = TASK_ENV.S_right.to(trajectory_unnormed.device)
    M_right = TASK_ENV.M_right.to(trajectory_unnormed.device)
    left_arm_dim = S_left.shape[-1]
    right_arm_dim = S_right.shape[-1]

    left_shoe = TASK_ENV.left_shoe
    right_shoe = TASK_ENV.right_shoe

    left_shoe_pos = torch.tensor(TASK_ENV.left_shoe.get_functional_point(0), device=trajectory_unnormed.device)[:3]
    right_shoe_pos = torch.tensor(TASK_ENV.right_shoe.get_functional_point(0), device=trajectory_unnormed.device)[:3]

    q_gripper_left = trajectory_unnormed[0, :, left_arm_dim].clone().detach()
    q_gripper_right = trajectory_unnormed[0, :, left_arm_dim + right_arm_dim + 1].clone().detach()

    actions = trajectory_unnormed[0]
    with torch.no_grad():
        left_gripper_positions, left_gripper_directions = get_gripper_keypoints(S_left, M_left, actions[:,:7], TASK_ENV.robot.left_gripper_bias)
        right_gripper_positions, right_gripper_directions = get_gripper_keypoints(S_right, M_right, actions[:,7:], TASK_ENV.robot.right_gripper_bias)

    def object_in_hand(object_pos, gripper_positions, threshold=0.03):
        time_step = 0
        return torch.norm(object_pos - gripper_positions["forward"][time_step], p=2) < threshold

    if TASK_ENV.FRAME_IDX == 0:
        TASK_ENV.guidance_state_recorder = GuidanceStateRecorder()

        TASK_ENV.guidance_state_recorder.l_shoe_pos_org = left_shoe_pos.clone().detach()
        TASK_ENV.guidance_state_recorder.r_shoe_pos_org = right_shoe_pos.clone().detach()

        TASK_ENV.guidance_state_recorder.l_poking = True
        TASK_ENV.guidance_state_recorder.r_poking = True
    
        TASK_ENV.guidance_state_recorder.l_rotate_grip = False
        TASK_ENV.guidance_state_recorder.r_rotate_grip = False

        TASK_ENV.guidance_state_recorder.l_rotate_put_in = False
        TASK_ENV.guidance_state_recorder.r_rotate_put_in = False

        TASK_ENV.guidance_state_recorder.r_release_grip = False

        height_shoe_box = (np.array(TASK_ENV.shoe_box.config['extents']) * np.array(TASK_ENV.shoe_box.config['scale']))[1] + TASK_ENV.shoe_box.get_pose().p[2]
        shoe_box_release_height_threshold = height_shoe_box + 0.2
        TASK_ENV.guidance_state_recorder.shoe_box_release_height_threshold = shoe_box_release_height_threshold
        

        TASK_ENV.guidance_state_recorder.stage = 0

    if TASK_ENV.guidance_state_recorder.stage == 0:
        left_shoe_in_short_range =  object_in_hand(left_shoe_pos, left_gripper_positions, threshold=0.20)
        right_shoe_in_short_range = object_in_hand(right_shoe_pos, right_gripper_positions, threshold=0.20)
        if left_shoe_in_short_range or right_shoe_in_short_range:
            TASK_ENV.guidance_state_recorder.l_rotate_grip = True
            TASK_ENV.guidance_state_recorder.r_rotate_grip = True

            TASK_ENV.guidance_state_recorder.stage = 1

    if TASK_ENV.guidance_state_recorder.stage == 1:
        if q_gripper_left[0] < 0.1 and q_gripper_right[0] < 0.1:
            left_shoe_in_hand =  object_in_hand(left_shoe_pos, left_gripper_positions, threshold=0.08)
            right_shoe_in_hand = object_in_hand(right_shoe_pos, right_gripper_positions, threshold=0.08)

            if left_shoe_in_hand and right_shoe_in_hand:
                TASK_ENV.guidance_state_recorder.l_poking = False
                TASK_ENV.guidance_state_recorder.r_poking = False
                TASK_ENV.guidance_state_recorder.l_rotate_grip = False
                TASK_ENV.guidance_state_recorder.r_rotate_grip = False
                TASK_ENV.guidance_state_recorder.stage = 2

    if TASK_ENV.guidance_state_recorder.stage == 2:
        if q_gripper_left[0] < 0.1 and q_gripper_right[0] < 0.1:
            left_shoe_in_hand =  object_in_hand(left_shoe_pos, left_gripper_positions, threshold=0.08)
            right_shoe_in_hand = object_in_hand(right_shoe_pos, right_gripper_positions, threshold=0.08)

            left_shoe_height_diff = left_gripper_positions["forward"][0,2] - TASK_ENV.guidance_state_recorder.l_shoe_pos_org[2]
            right_shoe_height_diff = right_gripper_positions["forward"][0,2] - TASK_ENV.guidance_state_recorder.r_shoe_pos_org[2]
            left_shoe_lifted = left_shoe_height_diff > 0.02 and left_shoe_in_hand
            right_shoe_lifted = right_shoe_height_diff > 0.02 and right_shoe_in_hand

            if left_shoe_lifted and right_shoe_lifted:
                TASK_ENV.guidance_state_recorder.stage = 3

    if TASK_ENV.guidance_state_recorder.stage == 3:
        if q_gripper_left[0] < 0.1:
            left_shoe_in_hand =  object_in_hand(left_shoe_pos, left_gripper_positions, threshold=0.08)
            distance_threshold_xy = 0.2
            left_shoe_target_pos = torch.tensor(TASK_ENV.shoe_box.get_functional_point(0), device=trajectory_unnormed.device)[:3]
            left_gripper_to_shoebox_dist_xy = torch.norm(left_gripper_positions["forward"][:,:2] - left_shoe_target_pos[:2], p=2, dim=1)

            if left_shoe_in_hand and (left_gripper_to_shoebox_dist_xy < distance_threshold_xy).sum()>3:
                TASK_ENV.guidance_state_recorder.l_rotate_put_in = True
                TASK_ENV.guidance_state_recorder.stage = 4

    if TASK_ENV.guidance_state_recorder.stage == 4:
        if q_gripper_left[0] > 0.5:
            TASK_ENV.guidance_state_recorder.l_rotate_put_in = False
            TASK_ENV.guidance_state_recorder.stage = 5

    if TASK_ENV.guidance_state_recorder.stage == 5:
        if q_gripper_right[0] < 0.1:
            right_shoe_in_hand =  object_in_hand(right_shoe_pos, right_gripper_positions, threshold=0.08)
            distance_threshold_xy = 0.2
            right_shoe_target_pos = torch.tensor(TASK_ENV.shoe_box.get_functional_point(1), device=trajectory_unnormed.device)[:3]
            right_gripper_to_shoebox_dist_xy = torch.norm(right_gripper_positions["forward"][:,:2] - right_shoe_target_pos[:2], p=2, dim=1)
            if right_shoe_in_hand and (right_gripper_to_shoebox_dist_xy < distance_threshold_xy).sum()>3:
                TASK_ENV.guidance_state_recorder.r_rotate_put_in = True
                TASK_ENV.guidance_state_recorder.stage = 6

    if TASK_ENV.guidance_state_recorder.stage == 6:

        distance_threshold_xy = 0.04
        right_shoe_target_pos = torch.tensor(TASK_ENV.shoe_box.get_functional_point(1), device=trajectory_unnormed.device)[:3]
        right_gripper_to_shoebox_dist_xy = torch.norm(right_gripper_positions["forward"][:,:2] - right_shoe_target_pos[:2], p=2, dim=1)

        use_release_right_shoebox_strict_flag = ((right_gripper_to_shoebox_dist_xy < distance_threshold_xy).sum()>3)
        if use_release_right_shoebox_strict_flag:
            TASK_ENV.guidance_state_recorder.r_release_grip = True
            TASK_ENV.guidance_state_recorder.stage = 7

    left_shoe_quat = left_shoe.get_functional_point(0)[3:]
    left_shoe_quat_mat = t3d.quaternions.quat2mat(left_shoe_quat)
    left_shoe_forward_direction = torch.tensor(left_shoe_quat_mat[:,1], device=trajectory_unnormed.device)

    right_shoe_quat = right_shoe.get_functional_point(0)[3:]
    right_shoe_quat_mat = t3d.quaternions.quat2mat(right_shoe_quat)
    right_shoe_forward_direction = torch.tensor(right_shoe_quat_mat[:,1], device=trajectory_unnormed.device)

    left_shoebox_target_quat = TASK_ENV.shoe_box.get_functional_point(0)[3:]
    left_shoebox_target_quat_mat = t3d.quaternions.quat2mat(left_shoebox_target_quat)
    left_shoebox_target_forward = torch.tensor(left_shoebox_target_quat_mat[:,1], device=trajectory_unnormed.device)

    right_shoebox_target_quat = TASK_ENV.shoe_box.get_functional_point(1)[3:]
    right_shoebox_target_quat_mat = t3d.quaternions.quat2mat(right_shoebox_target_quat)
    right_shoebox_target_forward = torch.tensor(right_shoebox_target_quat_mat[:,1], device=trajectory_unnormed.device)

    shoe_box_release_height_threshold = TASK_ENV.guidance_state_recorder.shoe_box_release_height_threshold

    TASK_ENV.guidance_state_recorder.l_rotate_put_in = False
    TASK_ENV.guidance_state_recorder.r_rotate_put_in = False

    env_info = {
        'l_poking': TASK_ENV.guidance_state_recorder.l_poking,
        'r_poking': TASK_ENV.guidance_state_recorder.r_poking,
        'l_rotate_grip': TASK_ENV.guidance_state_recorder.l_rotate_grip,
        'r_rotate_grip': TASK_ENV.guidance_state_recorder.r_rotate_grip,
        'l_rotate_put_in': TASK_ENV.guidance_state_recorder.l_rotate_put_in,
        'r_rotate_put_in': TASK_ENV.guidance_state_recorder.r_rotate_put_in,
        'r_release_grip': TASK_ENV.guidance_state_recorder.r_release_grip,
        'left_shoe_pos': left_shoe_pos.to('cpu'),
        'right_shoe_pos': right_shoe_pos.to('cpu'),
        'left_shoe_forward_direction': left_shoe_forward_direction.to('cpu'),
        'right_shoe_forward_direction': right_shoe_forward_direction.to('cpu'),
        'left_shoebox_target_forward': left_shoebox_target_forward.to('cpu'),
        'right_shoebox_target_forward': right_shoebox_target_forward.to('cpu'),
        'shoe_box_release_height_threshold': shoe_box_release_height_threshold,
        'stage': TASK_ENV.guidance_state_recorder.stage,
    }
    convert_to_tensors(env_info, return_tensors)
    return env_info


