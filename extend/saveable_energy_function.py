
import torch
import numpy as np
import torch.nn.functional as F
from typing import Any, Dict, Union
from extend.forward_kinematics_batch import get_gripper_keypoints_batch
from extend.safe_constraint_batch import (grippers_poking_cost_batch, behavior_alignment_cost_batch, grippers_rotation_cost_batch, grippers_distance_cost_batch)

def get_energy_by_task_name(task_name):
    if task_name == 'pick_dual_bottles':
        return energy_pick_dual_bottles
    elif task_name == 'pick_diverse_bottles':
        return energy_pick_dual_bottles
    elif task_name == 'handover_apple':
        return energy_handover_apple
    elif task_name == 'handover_block':
        return energy_handover_block
    elif task_name == 'stack_blocks_two':
        return energy_stack_blocks_two
    elif task_name == 'pour_water_to_cup':
        return energy_pour_water_to_cup
    elif task_name == 'place_dual_shoes':
        return energy_place_dual_shoes
    else:
        raise ValueError(f"Unsupported energy task name: {task_name}")

def get_env_keys_by_task_name(task_name):
    if task_name == 'pick_dual_bottles' or task_name == 'pick_diverse_bottles':
        return ['bottle1_functional_point', 'bottle2_functional_point']
    elif task_name == 'handover_apple':
        return ['apple_pos', 'l_poking', 'r_poking']
    elif task_name == 'handover_block':
        return ['box_pos_top', 'box_pos_bot', 'l_poking', 'r_distance']
    elif task_name == 'stack_blocks_two':
        return ['block1_pos', 'block2_pos', 'block1_dirs', 'block2_dirs', 'b1_left', 'b2_left', 'b1_poking', 'b2_poking', 'b1_alignment', 'b2_alignment', 'table_z_bias']
    elif task_name == 'pour_water_to_cup':
        return ['l_poking', 'r_poking', 'l_tilt', 'bottle_pos', 'cup_pos', 'bottle_top_to_functional_offset', 'cup_top_to_functional_offset', 'stage']
    elif task_name == 'place_dual_shoes':
        return ['l_poking', 'r_poking', 'l_rotate_grip', 'r_rotate_grip', 'l_rotate_put_in', 'r_rotate_put_in', 'r_release_grip', 'left_shoe_pos', 'right_shoe_pos', 'left_shoe_forward_direction', 'right_shoe_forward_direction', 'left_shoebox_target_forward', 'right_shoebox_target_forward', 'shoe_box_release_height_threshold', 'stage']
    else:
        raise ValueError(f"Unsupported energy task name: {task_name}")
    
def energy_pick_dual_bottles(
    env_meta: Dict[str, Union[torch.Tensor, float]],
    env_state: Dict[str, Union[torch.Tensor, float]],  
    trajectory_unnormed: torch.Tensor,  # [1, 8, 14]
    device: torch.device = None
):
    # env meta: static information
    # tensors
    S_left = env_meta['S_left'].to(device) # torch.Size([6, 6])
    M_left = env_meta['M_left'].to(device) # torch.Size([4, 4])
    S_right = env_meta['S_right'].to(device) # torch.Size([6, 6])
    M_right = env_meta['M_right'].to(device) # torch.Size([4, 4])
    # floats
    left_gripper_bias = env_meta['left_gripper_bias'] # float
    right_gripper_bias = env_meta['right_gripper_bias'] # float
    
    # env state: dynamic information
    bottle1_pos = env_state['bottle1_functional_point'][:3].to(device) # torch.Size([3])
    bottle2_pos = env_state['bottle2_functional_point'][:3].to(device) # torch.Size([3])

    actions = trajectory_unnormed[0] # torch.Size([8, 14])

    left_arm_dim = S_left.shape[-1] # 6
    right_arm_dim = S_right.shape[-1] # 6
    q_gripper_left = actions[:, left_arm_dim] # torch.Size([8])
    q_gripper_right = actions[:, left_arm_dim + right_arm_dim + 1] # torch.Size([8])
   
    smooth_l1_loss = torch.nn.SmoothL1Loss(reduction='none', beta=0.1)
    distance_process = lambda x: smooth_l1_loss(x, torch.zeros_like(x))

    left_gripper_positions, left_gripper_directions = get_gripper_keypoints_batch(S_left, M_left, actions[:,:7], left_gripper_bias)
    right_gripper_positions, right_gripper_directions = get_gripper_keypoints_batch(S_right, M_right, actions[:,7:], right_gripper_bias)

    # ===== poking cost (batch) =====
    cost3_l = grippers_poking_cost_batch(
        bottle1_pos,                                    # [3]
        left_gripper_positions["forward"],              # [N,3]
        left_gripper_directions["forward"]              # [N,3]
    )                                                   # [N]

    cost3_r = grippers_poking_cost_batch(
        bottle2_pos,                                    # [3]
        right_gripper_positions["forward"],             # [N,3]
        right_gripper_directions["forward"]             # [N,3]
    )                                                   # [N]

    smooth_l1 = torch.nn.SmoothL1Loss(reduction="none", beta=0.1)
    distance_process = lambda x: smooth_l1(x, torch.zeros_like(x))

    cost3_l = distance_process(cost3_l) * q_gripper_left.detach()
    cost3_r = distance_process(cost3_r) * q_gripper_right.detach()

    loss = cost3_l.sum() + cost3_r.sum()

    loss_info = {
        "cost3_l": [round(x.item(), 4) for x in cost3_l],
        "cost3_r": [round(x.item(), 4) for x in cost3_r],
    }

    return loss, loss_info

def energy_pour_water_to_cup(
    env_meta: Dict[str, Union[torch.Tensor, float]],
    env_state: Dict[str, Union[torch.Tensor, float]],
    trajectory_unnormed: torch.Tensor,  # [1, N, 14]
    device: torch.device = None
):
    """
    trajectory_unnormed: [1, N, 14]
    """
    # ===== env meta =====
    S_left  = env_meta['S_left'].to(device)
    M_left  = env_meta['M_left'].to(device)
    S_right = env_meta['S_right'].to(device)
    M_right = env_meta['M_right'].to(device)

    left_gripper_bias  = env_meta['left_gripper_bias']
    right_gripper_bias = env_meta['right_gripper_bias']

    # ===== env state =====
    l_poking = env_state['l_poking']
    r_poking = env_state['r_poking']

    bottle_pos = env_state['bottle_pos'].to(device)  # [3]
    cup_pos    = env_state['cup_pos'].to(device)     # [3]

    # ===== dims / actions =====
    actions = trajectory_unnormed[0]  # [N, 14]
    N = actions.shape[0]

    left_arm_dim  = S_left.shape[-1]   # 6
    right_arm_dim = S_right.shape[-1]  # 6

    # gripper open(1)/close(0)
    q_gripper_left  = trajectory_unnormed[0, :, left_arm_dim]                       # [N]
    q_gripper_right = trajectory_unnormed[0, :, left_arm_dim + right_arm_dim + 1]   # [N]

    # distance process
    smooth_l1_loss = torch.nn.SmoothL1Loss(reduction='none', beta=0.1)
    distance_process = lambda x: smooth_l1_loss(x, torch.zeros_like(x))

    loss = trajectory_unnormed.sum() * 0.0
    loss_info: Dict[str, list] = {}

    left_pos, left_dir = get_gripper_keypoints_batch(
        S_left, M_left, actions[:, :7], left_gripper_bias
    )
    right_pos, right_dir = get_gripper_keypoints_batch(
        S_right, M_right, actions[:, 7:], right_gripper_bias
    )

    if l_poking or r_poking:
        cost3_l_raw = grippers_poking_cost_batch(
            bottle_pos,                 # [3]
            left_pos["forward"],        # [N,3]
            left_dir["forward"]         # [N,3]
        )                               # [N]

        cost3_r_raw = grippers_poking_cost_batch(
            cup_pos,
            right_pos["forward"],
            right_dir["forward"]
        )                               # [N]

        cost3_l = distance_process(cost3_l_raw) * q_gripper_left.detach()    # [N]
        cost3_r = distance_process(cost3_r_raw) * q_gripper_right.detach()   # [N]

        loss = loss + cost3_l.sum() * float(l_poking) + cost3_r.sum() * float(r_poking)

        loss_info.update({
            "l_poking": [round((x * float(l_poking)).item(), 4) for x in cost3_l_raw],
            "r_poking": [round((x * float(r_poking)).item(), 4) for x in cost3_r_raw],
        })

    return loss, loss_info

def energy_place_dual_shoes(
    env_meta: Dict[str, Union[torch.Tensor, float]],
    env_state: Dict[str, Union[torch.Tensor, float]],
    trajectory_unnormed: torch.Tensor,  # [1, N, 14]
    device: torch.device = None
):
    """
    trajectory_unnormed: [1, N, 14]
    """
    # ===== env meta =====
    S_left  = env_meta['S_left'].to(device)
    M_left  = env_meta['M_left'].to(device)
    S_right = env_meta['S_right'].to(device)
    M_right = env_meta['M_right'].to(device)
    left_gripper_bias  = env_meta['left_gripper_bias']
    right_gripper_bias = env_meta['right_gripper_bias']

    # ===== env state =====
    l_poking = env_state['l_poking']
    r_poking = env_state['r_poking']
    l_rotate_grip = env_state['l_rotate_grip']
    r_rotate_grip = env_state['r_rotate_grip']
    l_rotate_put_in = env_state['l_rotate_put_in']
    r_rotate_put_in = env_state['r_rotate_put_in']
    r_release_grip = env_state['r_release_grip']

    left_shoe_pos  = env_state['left_shoe_pos'].to(device).double()   # [3]
    right_shoe_pos = env_state['right_shoe_pos'].to(device).double()  # [3]

    left_shoe_forward_direction  = env_state['left_shoe_forward_direction'].to(device).double()   # [3]
    right_shoe_forward_direction = env_state['right_shoe_forward_direction'].to(device).double()  # [3]

    left_shoebox_target_forward  = env_state['left_shoebox_target_forward'].to(device).double()   # [3]
    right_shoebox_target_forward = env_state['right_shoebox_target_forward'].to(device).double()  # [3]

    shoe_box_release_height_threshold = env_state['shoe_box_release_height_threshold']

    # ===== dims / actions =====
    actions = trajectory_unnormed[0]  # [N,14]
    N = actions.shape[0]
    left_arm_dim  = S_left.shape[-1]   # 6
    right_arm_dim = S_right.shape[-1]  # 6

    # gripper open(1)/close(0)
    q_gripper_left  = trajectory_unnormed[0, :, left_arm_dim].clone().detach()                    # [N]
    q_gripper_right = trajectory_unnormed[0, :, left_arm_dim + right_arm_dim + 1].clone().detach()# [N]

    smooth_l1_loss = torch.nn.SmoothL1Loss(reduction='none', beta=0.1)
    distance_process = lambda x: smooth_l1_loss(x, torch.zeros_like(x))

    loss = trajectory_unnormed.sum() * 0.0
    loss_info: Dict[str, list] = {}

    # ============================================================
    # 时间权重（并行）
    # 原逻辑：从 start_index 开始线性升高，并 clamp 到 lower_bound
    # ============================================================
    lower_bound = 0.1
    upper_bound = 0.75
    start_index = 2

    t = torch.arange(N, device=trajectory_unnormed.device, dtype=trajectory_unnormed.dtype)  # [N]
    denom = max(N - start_index - 1, 1)
    weight_time_tensor = (t - start_index) / denom * (upper_bound - lower_bound) + lower_bound
    weight_time_tensor = torch.clamp(weight_time_tensor, min=lower_bound)  # [N]

    # ============================================================
    # FK（批量一次算好，后面各项共享）
    # ============================================================
    left_gripper_positions, left_gripper_directions = get_gripper_keypoints_batch(
        S_left, M_left, actions[:, :7], left_gripper_bias
    )
    right_gripper_positions, right_gripper_directions = get_gripper_keypoints_batch(
        S_right, M_right, actions[:, 7:], right_gripper_bias
    )
    # forward/top: [N,3]

    # ============================================================
    # 1) poking（并行）
    # ============================================================
    if l_poking or r_poking:
        # print('Computing poking loss...')
        cost3_l_raw = grippers_poking_cost_batch(
            left_shoe_pos,                               # [3]
            left_gripper_positions["forward"],           # [N,3]
            left_gripper_directions["forward"],          # [N,3]
        )                                                # [N]
        cost3_r_raw = grippers_poking_cost_batch(
            right_shoe_pos,
            right_gripper_positions["forward"],
            right_gripper_directions["forward"],
        )                                                # [N]

        cost3_l = distance_process(cost3_l_raw) * q_gripper_left      # [N]
        cost3_r = distance_process(cost3_r_raw) * q_gripper_right     # [N]

        cost3 = cost3_l.sum() * float(l_poking) + cost3_r.sum() * float(r_poking)
        loss = loss + cost3


        loss_info.update({
            "l_poking": [round((x * float(l_poking)).item(), 4) for x in cost3_l_raw],
            "r_poking": [round((x * float(r_poking)).item(), 4) for x in cost3_r_raw],
        })

    # ============================================================
    # 2) rotate_grip（本来就 batch，这里主要把 FK 调用换成 batch）
    #    仅最后关节参与角度对齐梯度：actions_grad_masked
    # ============================================================
    if l_rotate_grip or r_rotate_grip:
        # print('Computing rotate_grip loss...')
        last_action_left  = actions[:, 5]   # [N]
        last_action_right = actions[:, 12]  # [N]

        actions_grad_masked = actions.clone().detach()  # [N,14]
        actions_grad_masked[:, 5]  = last_action_left
        actions_grad_masked[:, 12] = last_action_right

        left_pos_m, left_dir_m = get_gripper_keypoints_batch(
            S_left, M_left, actions_grad_masked[:, :7], left_gripper_bias
        )
        right_pos_m, right_dir_m = get_gripper_keypoints_batch(
            S_right, M_right, actions_grad_masked[:, 7:], right_gripper_bias
        )

        def _process_angle_wrap(angle_tensor: torch.Tensor) -> torch.Tensor:
            """
            angle_tensor: [N]
            """
            cross = (angle_tensor > np.pi / 2).any() and (angle_tensor < -np.pi / 2).any()
            if cross:
                angle_tensor = torch.where(angle_tensor < 0, angle_tensor + 2 * np.pi, angle_tensor)
            return angle_tensor

        def gripper_to_shoe_near_direction(
            shoe_forward_direction: torch.Tensor,   # [3]
            gripper_directions: Dict[str, torch.Tensor], # each [N,3]
            q_gripper: torch.Tensor,                # [N]
            actions: torch.Tensor,                  # [N,14] (unused but kept for signature compatibility)
            weight_time_tensor: torch.Tensor,        # [N] (unused in your current logic)
            device
        ):
            gripper_up   = gripper_directions["top"].double()      # [N,3]
            gripper_down = (-gripper_directions["top"]).double()   # [N,3]

            target = shoe_forward_direction.double()[None, :].expand(N, -1)  # [N,3]

            delta_top  = angle_diff_batch(target, gripper_up)    # [N]
            delta_top  = _process_angle_wrap(delta_top)

            delta_down = angle_diff_batch(target, gripper_down)  # [N]
            delta_down = _process_angle_wrap(delta_down)

            # 用 sum 更小的那一支（标量分支，没必要强行向量化）
            if delta_top.sum() < delta_down.sum():
                delta = delta_top
            else:
                delta = delta_down

            cost_list = activate_angle_loss(delta) * q_gripper  # [N]  (你原代码已经去掉了 weight_time_tensor)
            return cost_list.sum(), [round(x.item(), 4) for x in cost_list]

        cost_rotate_alpha = 0.1

        if l_rotate_grip:
            _loss, _series = gripper_to_shoe_near_direction(
                left_shoe_forward_direction, left_dir_m, q_gripper_left, actions, weight_time_tensor, device
            )
            loss = loss + _loss * cost_rotate_alpha
            loss_info["l_rotate_grip"] = _series

        if r_rotate_grip:
            _loss, _series = gripper_to_shoe_near_direction(
                right_shoe_forward_direction, right_dir_m, q_gripper_right, actions, weight_time_tensor, device
            )
            loss = loss + _loss * cost_rotate_alpha
            loss_info["r_rotate_grip"] = _series
        

        # 加一个功能轴对齐的损失
        # 直接规定为向上
        left_shoe_functional_dir = torch.tensor([0.0, 0.0, 1.0], device=trajectory_unnormed.device).double()
        right_shoe_functional_dir = torch.tensor([0.0, 0.0, 1.0], device=trajectory_unnormed.device).double()

        cost_align_l_list = behavior_alignment_cost_batch(
            left_shoe_functional_dir,
            left_gripper_positions["forward"],
            left_shoe_pos,
            keypoints_offset=0.03,
            lambda_param=1.0
        )
        cost_align_r_list = behavior_alignment_cost_batch(
            right_shoe_functional_dir,
            right_gripper_positions["forward"],
            right_shoe_pos,
            keypoints_offset=0.03,
            lambda_param=1.0
        )

        cost_align_l = distance_process(cost_align_l_list)  * q_gripper_left
        cost_align_r = distance_process(cost_align_r_list)  * q_gripper_right

        alpha_align = 1
        if l_rotate_grip:
            loss = loss + cost_align_l.sum() * alpha_align
            loss_info['l_align'] = [round((x * l_poking).item(), 4) for x in cost_align_l_list]
        if r_rotate_grip:
            loss = loss + cost_align_r.sum() * alpha_align
            loss_info['r_align'] = [round((x * r_poking).item(), 4) for x in cost_align_r_list]


    # ============================================================
    # 3) rotate_put_in（本来就 batch，这里只去掉 repeat + 保持并行）
    # ============================================================
    if l_rotate_put_in or r_rotate_put_in:
        # print('Computing rotate_put_in loss...')
         # 夹具 动作：1 打开  0 闭合
        q_gripper_shoebox_left  = trajectory_unnormed[0, :, left_arm_dim].clone().detach()                     # [N]
        q_gripper_shoebox_right = trajectory_unnormed[0, :, left_arm_dim + right_arm_dim + 1].clone().detach() # [N]

        q_gripper_shoebox_left = torch.where(q_gripper_shoebox_left < 0.5, torch.zeros_like(q_gripper_shoebox_left), q_gripper_shoebox_left)
        q_gripper_shoebox_right = torch.where(q_gripper_shoebox_right < 0.5, torch.zeros_like(q_gripper_shoebox_right), q_gripper_shoebox_right)

        q_use_left  = 1.0 - q_gripper_shoebox_left   # [N] 闭合才施加
        q_use_right = 1.0 - q_gripper_shoebox_right  # [N]

        def rotate_toward_shoebox(
            target_forward: torch.Tensor,                 # [3]
            gripper_dirs: Dict[str, torch.Tensor],         # each [N,3]
            q_use: torch.Tensor,                           # [N]
            weight_time_tensor: torch.Tensor,              # [N]
        ):
            gripper_up = gripper_dirs["top"].double()      # [N,3]
            target = target_forward.double()[None, :].expand(N, -1)  # [N,3]
            delta = angle_diff_batch(target, gripper_up)   # [N]

            cross = (delta > np.pi / 2).any() and (delta < -np.pi / 2).any()
            if cross:
                delta = torch.where(delta < 0, delta + 2 * np.pi, delta)

            cost_list = activate_angle_loss(delta) * weight_time_tensor * q_use  # [N]
            return cost_list.sum(), [round(x.item(), 4) for x in cost_list]

        cost_rotate_alpha = 1.0

        if l_rotate_put_in:
            _loss, _series = rotate_toward_shoebox(
                left_shoebox_target_forward, left_gripper_directions, q_use_left, weight_time_tensor
            )
            loss = loss + _loss * cost_rotate_alpha
            loss_info["l_rotate_put_in"] = _series

        if r_rotate_put_in:
            _loss, _series = rotate_toward_shoebox(
                right_shoebox_target_forward, right_gripper_directions, q_use_right, weight_time_tensor
            )
            loss = loss + _loss * cost_rotate_alpha
            loss_info["r_rotate_put_in"] = _series

    # ============================================================
    # 4) r_release_grip（去 loop：mask + 向量化 time weight）
    # ============================================================
    if r_release_grip:
        # print('Computing r_release_grip loss...')
        q_gripper_shoebox_right = trajectory_unnormed[0, :, left_arm_dim + right_arm_dim + 1]  # [N]

        right_z = right_gripper_positions["forward"][:, 2]  # [N]
        mask = (right_z < shoe_box_release_height_threshold).to(q_gripper_shoebox_right.dtype)  # [N]

        # 原逻辑：weight_time = max((t-1)/N, 1/N)
        tt = torch.arange(N, device=trajectory_unnormed.device, dtype=q_gripper_shoebox_right.dtype)  # [N]
        w = torch.clamp((tt - 1.0) / float(N), min=1.0 / float(N))  # [N]

        # 原逻辑：低于阈值时 cost = -q_gripper * weight
        cost_release_right = (-q_gripper_shoebox_right) * w * mask   # [N]

        cost_rotate_alpha = 10.0
        loss = loss + cost_release_right.sum() * cost_rotate_alpha
        loss_info["r_release_grip"] = [round(x.item(), 4) for x in cost_release_right]

    return loss, loss_info

def energy_handover_apple(
        env_meta: Dict[str, Union[torch.Tensor, float]],
        env_state: Dict[str, Union[torch.Tensor, float]],    
        trajectory_unnormed: torch.Tensor, 
        device: torch.device = None
        ):
   # env meta: static information
    # tensors
    S_left = env_meta['S_left'].to(device)
    M_left = env_meta['M_left'].to(device)
    S_right = env_meta['S_right'].to(device)
    M_right = env_meta['M_right'].to(device)
    # floats
    left_gripper_bias = env_meta['left_gripper_bias']
    right_gripper_bias = env_meta['right_gripper_bias']
    
    # env state: dynamic information
    apple_pos = env_state['apple_pos'].to(device)
    l_poking = env_state['l_poking']
    r_poking = env_state['r_poking']

    actions = trajectory_unnormed[0]

    left_arm_dim = S_left.shape[-1]
    right_arm_dim = S_right.shape[-1]
    q_gripper_left = actions[:, left_arm_dim]
    q_gripper_right = actions[:, left_arm_dim + right_arm_dim + 1]
   
    smooth_l1_loss = torch.nn.SmoothL1Loss(reduction='none', beta=0.1)
    distance_process = lambda x: smooth_l1_loss(x, torch.zeros_like(x))

    left_gripper_positions, left_gripper_directions = get_gripper_keypoints_batch(S_left, M_left, actions[:,:7], left_gripper_bias)
    right_gripper_positions, right_gripper_directions = get_gripper_keypoints_batch(S_right, M_right, actions[:,7:], right_gripper_bias)

    t = torch.arange(actions.shape[0], device=device, dtype=actions.dtype)

    weights = torch.where(
        t > 1,
        (t - 1) / 6.0,
        torch.full_like(t, 1.0 / 6.0)
    )  # [N]

    cost3_l = float(l_poking) * weights * grippers_poking_cost_batch(
        apple_pos,
        left_gripper_positions["forward"],     # [N,3]
        left_gripper_directions["forward"]      # [N,3]
    )                            # [N]

    cost3_r = float(r_poking) * weights * grippers_poking_cost_batch(
        apple_pos,
        right_gripper_positions["forward"],
        right_gripper_directions["forward"]
    )                            # [N]

    cost3_l = distance_process(cost3_l) * q_gripper_left.detach()
    cost3_r = distance_process(cost3_r) * q_gripper_right.detach()

    loss = cost3_l.sum() + cost3_r.sum()

    loss_info = {
        "cost3_l": [round(x.item(), 4) for x in cost3_l],
        "cost3_r": [round(x.item(), 4) for x in cost3_r],
    }

    return loss, loss_info

def energy_handover_block(
        env_meta: Dict[str, Union[torch.Tensor, float]],
        env_state: Dict[str, Union[torch.Tensor, float]],    
        trajectory_unnormed: torch.Tensor, 
        device: torch.device = None
        ):
   # env meta: static information
    # tensors
    S_left = env_meta['S_left'].to(device)
    M_left = env_meta['M_left'].to(device)
    S_right = env_meta['S_right'].to(device)
    M_right = env_meta['M_right'].to(device)
    # floats
    left_gripper_bias = env_meta['left_gripper_bias']
    right_gripper_bias = env_meta['right_gripper_bias']
    
    # env state: dynamic information
    box_pos_top = env_state['box_pos_top'].to(device)
    box_pos_bot = env_state['box_pos_bot'].to(device)
    l_poking = env_state['l_poking']
    r_distance = env_state['r_distance']

    actions = trajectory_unnormed[0]

    left_arm_dim = S_left.shape[-1]
    right_arm_dim = S_right.shape[-1]
    q_gripper_left = actions[:, left_arm_dim]
    q_gripper_right = actions[:, left_arm_dim + right_arm_dim + 1]
   
    smooth_l1_loss = torch.nn.SmoothL1Loss(reduction='none', beta=0.1)
    distance_process = lambda x: smooth_l1_loss(x, torch.zeros_like(x))

    left_gripper_positions, left_gripper_directions = get_gripper_keypoints_batch(S_left, M_left, actions[:,:7], left_gripper_bias)
    right_gripper_positions, right_gripper_directions = get_gripper_keypoints_batch(S_right, M_right, actions[:,7:], right_gripper_bias)

    t = torch.arange(actions.shape[0], device=device, dtype=actions.dtype)  # [N]
    time_weights = t + 1.0                                                  # [N]

    cost3_l = float(l_poking) * grippers_poking_cost_batch(
        box_pos_top,
        left_gripper_positions["forward"],      # [N,3]
        left_gripper_directions["forward"]      # [N,3]
    )                                           # [N]

    # DP base policies may keep the right gripper nearly stationary on this task.
    # This auxiliary distance term encourages right-hand motion toward the object.
    # For policies without this failure mode (e.g., DP3 or RDT), this term can be removed.
    cost3_r = (
        float(r_distance)
        * 3.0
        * time_weights
        * grippers_distance_cost_batch(
            box_pos_bot,
            right_gripper_positions["forward"]  # [N,3]
        )
    )                                           # [N]

    cost3_l = distance_process(cost3_l) * q_gripper_left.detach()
    cost3_r = distance_process(cost3_r) * q_gripper_right.detach()

    loss = cost3_l.sum() + cost3_r.sum()

    loss_info = {
        "cost3_l": [round(x.item(), 4) for x in cost3_l],
        "cost3_r": [round(x.item(), 4) for x in cost3_r],
    }

    return loss, loss_info

def energy_stack_blocks_two(
        env_meta: Dict[str, Union[torch.Tensor, float]],
        env_state: Dict[str, Union[torch.Tensor, float]],    
        trajectory_unnormed: torch.Tensor, # [1, 8, 14]
        device: torch.device = None
        ):
   # env meta: static information
    # tensors
    S_left = env_meta['S_left'].to(device)
    M_left = env_meta['M_left'].to(device)
    S_right = env_meta['S_right'].to(device)
    M_right = env_meta['M_right'].to(device)
    # floats
    left_gripper_bias = env_meta['left_gripper_bias']
    right_gripper_bias = env_meta['right_gripper_bias']
    
    # env state: dynamic information
    block1_pos = env_state['block1_pos'].to(device)
    block2_pos = env_state['block2_pos'].to(device)
    block1_dirs = env_state['block1_dirs'].to(device).to(torch.float64) # [4,3]
    block2_dirs = env_state['block2_dirs'].to(device).to(torch.float64) # [4,3]
    b1_left = env_state['b1_left']
    b2_left = env_state['b2_left']
    b1_poking = env_state['b1_poking']
    b2_poking = env_state['b2_poking']
    b1_alignment = env_state['b1_alignment']
    b2_alignment = env_state['b2_alignment']
    table_z_bias = env_state['table_z_bias']

    left_arm_dim = S_left.shape[-1]   # 6
    right_arm_dim = S_right.shape[-1] # 6

    smooth_l1_loss = torch.nn.SmoothL1Loss(reduction='none', beta=0.1)
    distance_process = lambda x: smooth_l1_loss(x, torch.zeros_like(x))

    actions = trajectory_unnormed[0]

    if b1_left:
        block1_gripper_positions, block1_gripper_directions = get_gripper_keypoints_batch(S_left, M_left, actions[:,:7], left_gripper_bias)
        actions_part_detach = actions.clone().detach()
        actions_part_detach[:,left_arm_dim - 1] = actions.clone()[:, left_arm_dim - 1]
        _, block1_gripper_directions_part_detach = get_gripper_keypoints_batch(S_left, M_left, actions_part_detach[:,:7], left_gripper_bias)
        q_gripper_block1 = actions[:, left_arm_dim]
    else:
        block1_gripper_positions, block1_gripper_directions = get_gripper_keypoints_batch(S_right, M_right, actions[:,7:], right_gripper_bias)
        actions_part_detach = actions.clone().detach()
        actions_part_detach[:,left_arm_dim + right_arm_dim] = actions.clone()[:, left_arm_dim + right_arm_dim]
        _, block1_gripper_directions_part_detach = get_gripper_keypoints_batch(S_right, M_right, actions_part_detach[:,7:], right_gripper_bias)
        q_gripper_block1 = actions[:, left_arm_dim + right_arm_dim + 1]

    if b2_left:
        block2_gripper_positions, block2_gripper_directions = get_gripper_keypoints_batch(S_left, M_left, actions[:,:7], left_gripper_bias)
        actions_part_detach = actions.clone().detach()
        actions_part_detach[:,left_arm_dim - 1] = actions.clone()[:, left_arm_dim - 1]
        _, block2_gripper_directions_part_detach = get_gripper_keypoints_batch(S_left, M_left, actions_part_detach[:,:7], left_gripper_bias)
        q_gripper_block2 = actions[:, left_arm_dim]
    else:
        block2_gripper_positions, block2_gripper_directions = get_gripper_keypoints_batch(S_right, M_right, actions[:,7:], right_gripper_bias)
        actions_part_detach = actions.clone().detach()
        actions_part_detach[:,left_arm_dim + right_arm_dim] = actions.clone()[:, left_arm_dim + right_arm_dim]
        _, block2_gripper_directions_part_detach = get_gripper_keypoints_batch(S_right, M_right, actions_part_detach[:,7:], right_gripper_bias)
        q_gripper_block2 = actions[:, left_arm_dim + right_arm_dim + 1]
    
    cost3_1 = float(b1_poking) * (
        grippers_poking_cost_batch(
            block1_pos,
            block1_gripper_positions["forward"],              # [N,3]
            block1_gripper_directions["forward"]              # [N,3]
        )
        +
        grippers_rotation_cost_batch(
            block1_gripper_directions_part_detach["top"],     # [N,3]
            block1_dirs                                       # [4,3]
        )
    )                                                         # [N]

    cost3_2 = float(b2_poking) * (
        grippers_poking_cost_batch(
            block2_pos,
            block2_gripper_positions["forward"],              # [N,3]
            block2_gripper_directions["forward"]              # [N,3]
        )
        +
        grippers_rotation_cost_batch(
            block2_gripper_directions_part_detach["top"],     # [N,3]
            block2_dirs                                       # [4,3]
        )
    )                                                         # [N]

    cost2_1 = float(b1_alignment) * behavior_alignment_cost_batch(torch.tensor([0.,0.,1.], device=device, dtype=torch.float64), 
                                        block1_gripper_positions["forward"], 
                                        torch.tensor([0, -0.13, 0.7525 + table_z_bias], device=device, dtype=torch.float64),
                                        0.002, 1.)
    cost2_2 = float(b2_alignment) * behavior_alignment_cost_batch(torch.tensor([0.,0.,1.], device=device, dtype=torch.float64), 
                                        block2_gripper_positions["forward"], 
                                        block1_pos + torch.tensor([0, 0, 0.05], device=device, dtype=torch.float64),
                                        0.002, 1.)

    cost3_1 = distance_process(cost3_1) * q_gripper_block1.detach()
    cost3_2 = distance_process(cost3_2) * q_gripper_block2.detach()

    cost2_1 = distance_process(cost2_1) * (1 - q_gripper_block1.detach())
    cost2_2 = distance_process(cost2_2) * (1 - q_gripper_block2.detach())

    loss = cost3_1.sum() + cost3_2.sum() + cost2_1.sum() + cost2_2.sum()

    loss_info = {
        "cost3_1": [round(x.item(), 4) for x in cost3_1],
        "cost3_2": [round(x.item(), 4) for x in cost3_2],
    }

    return loss, loss_info

def angle_diff_batch(u, v, eps=1e-8, cross_eps=1e-6):
    assert u.shape[-1] == 3 and v.shape[-1] == 3
    device, dtype = u.device, u.dtype
    u = F.normalize(u, dim=-1)
    v = F.normalize(v, dim=-1)
    uv_cross = torch.cross(u, v, dim=-1)
    uv_cross_norm = torch.linalg.norm(uv_cross, dim=-1, keepdim=True)
    n_auto = uv_cross / (uv_cross_norm + eps)
    n_fallback = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).expand_as(u)
    use_auto = (uv_cross_norm.squeeze(-1) >= cross_eps)
    n = torch.where(use_auto.unsqueeze(-1), n_auto, n_fallback)
    n = F.normalize(n, dim=-1)
    u_par = (u * n).sum(dim=-1, keepdim=True) * n
    v_par = (v * n).sum(dim=-1, keepdim=True) * n
    u_proj = u - u_par
    v_proj = v - v_par
    u_proj_norm = torch.linalg.norm(u_proj, dim=-1, keepdim=True)
    v_proj_norm = torch.linalg.norm(v_proj, dim=-1, keepdim=True)
    u_hat = torch.where(u_proj_norm > eps, u_proj / (u_proj_norm + eps), u_proj)
    v_hat = torch.where(v_proj_norm > eps, v_proj / (v_proj_norm + eps), v_proj)
    x = (u_hat * v_hat).sum(dim=-1).clamp(-1 + eps, 1 - eps)
    y = (torch.cross(u_hat, v_hat, dim=-1) * n).sum(dim=-1)
    theta = torch.atan2(y, x)
    degenerate_proj = (u_proj_norm.squeeze(-1) <= cross_eps) | (v_proj_norm.squeeze(-1) <= cross_eps)
    dot_uv = (u * v).sum(dim=-1)
    theta_fallback = torch.where(dot_uv >= 0, torch.zeros_like(dot_uv), torch.full_like(dot_uv, torch.pi))
    theta = torch.where(degenerate_proj, theta_fallback, theta)
    return theta

def activate_angle_loss(angle_tensor):
    smooth_l1_03 = torch.nn.SmoothL1Loss(reduction='none', beta=0.3)
    theta_process = lambda x: smooth_l1_03(x, torch.zeros_like(x))
    return theta_process(angle_tensor)