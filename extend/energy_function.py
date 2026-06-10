from extend.forward_kinematics import build_S_list_and_M_world

def init_SM(TASK_ENV):
    articulation = TASK_ENV.robot.left_entity
    ee_link_child = TASK_ENV.robot.left_ee.child_link 
    S_left, M_left, _ = build_S_list_and_M_world(articulation, ee_link_child)
    articulation = TASK_ENV.robot.right_entity
    ee_link_child = TASK_ENV.robot.right_ee.child_link
    S_right, M_right, _ = build_S_list_and_M_world(articulation, ee_link_child)
    TASK_ENV.S_left=S_left
    TASK_ENV.M_left=M_left
    TASK_ENV.S_right=S_right
    TASK_ENV.M_right=M_right

def get_guid_max_timesteps_by_policy_name(policy_name="DP"):
    if policy_name == "DP":
        return 3

def get_guid_param_by_task_name(task_name, policy_name="DP"):
    if policy_name=="DP":
        if task_name == 'pick_dual_bottles':
            return 1, 0.1
        elif task_name == 'pick_diverse_bottles':
            return 1, 0.01 
        elif task_name == 'handover_block':
            return 1, 0.1 
        elif task_name == 'handover_apple':
            return 1, 0.1 
        elif task_name == 'stack_blocks_two':
            return 1, 0.1 
        elif task_name == 'place_dual_shoes':
            return 1, 0.1
        elif task_name == 'pour_water_to_cup':
            return 5, 0.03
        else:
            raise ValueError(f"Unsupported energy task name: {task_name}")

