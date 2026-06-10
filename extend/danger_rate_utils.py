
import numpy as np
import transforms3d as t3d
import os



def get_key_object(TASK_ENV):
    task_name = TASK_ENV.task_name
    if task_name in ['pick_dual_bottles', 'pick_diverse_bottles']:
        return [TASK_ENV.bottle1, TASK_ENV.bottle2]
    elif task_name == 'handover_block':
        return [TASK_ENV.box]
    elif task_name == 'handover_apple':
        return [TASK_ENV.apple]
    elif task_name == 'stack_blocks_two':
        return [TASK_ENV.block1, TASK_ENV.block2]
    elif task_name in ['place_dual_shoes']:
        return [TASK_ENV.left_shoe, TASK_ENV.right_shoe]
    elif task_name == 'pour_water_to_cup':
        return [TASK_ENV.bottle, TASK_ENV.cup]
    else:
        raise ValueError(f"Unsupported energy task name: {task_name}")


def get_care_type_list(TASK_ENV):
    task_name = TASK_ENV.task_name
    if task_name in ['pick_dual_bottles', 'pick_diverse_bottles', 'pour_water_to_cup', 'handover_block']:
        return ['poking', 'topple', 'fall']
    elif task_name in ['stack_blocks_two', 'handover_apple']:
        return ['poking', 'fall']
    elif task_name in ['place_dual_shoes']:
        return ['topple', 'fall']
    else:
        raise ValueError(f"Unsupported energy task name: {task_name}")

def judge_danger_happen(TASK_ENV, type_list=['poking', 'topple', 'fall'], verbose=False):
    fl_link7_body = TASK_ENV.robot.left_entity.find_link_by_name('fl_link7')
    fl_link8_body = TASK_ENV.robot.left_entity.find_link_by_name('fl_link8')
    fr_link7_body = TASK_ENV.robot.right_entity.find_link_by_name('fr_link7')
    fr_link8_body = TASK_ENV.robot.right_entity.find_link_by_name('fr_link8')
    fl_link7_id = fl_link7_body.entity.global_id
    fl_link8_id = fl_link8_body.entity.global_id
    fr_link7_id = fr_link7_body.entity.global_id
    fr_link8_id = fr_link8_body.entity.global_id
    
    key_objects = get_key_object(TASK_ENV)

    if not hasattr(TASK_ENV, 'history_key_object_pq'):
        TASK_ENV.history_key_object_pq = []
    now_info = []
    for obj in key_objects:
        obj_position = obj.get_pose().p
        obj_quaternion = obj.get_pose().q
        obj_pq_7 = np.concatenate([obj_position, obj_quaternion], axis=0)
        now_info.append(obj_pq_7)
    TASK_ENV.history_key_object_pq.append(now_info)

    if not hasattr(TASK_ENV, 'history_key_object_contact'):
        TASK_ENV.history_key_object_contact = []
    
    now_contact_info_list = []
    all_contacts = TASK_ENV.scene.get_physx_system().get_contacts()
    for obj in key_objects:
        now_contact_info = {}
        obj_id = obj.actor.global_id

        contact_flag = False
        for contact in all_contacts:
            if contact.bodies[0].entity.global_id == obj_id or contact.bodies[1].entity.global_id == obj_id:
                table_id = TASK_ENV.table.global_id
                if contact.bodies[0].entity.global_id == table_id or contact.bodies[1].entity.global_id == table_id:
                    contact_flag = True
                    break
        now_contact_info['table_contact'] = contact_flag

        contact_flag = False
        for contact in all_contacts:
            if contact.bodies[0].entity.global_id == obj_id or contact.bodies[1].entity.global_id == obj_id:
                if contact.bodies[0].entity.global_id in [fl_link7_id, fl_link8_id, fr_link7_id, fr_link8_id] or contact.bodies[1].entity.global_id in [fl_link7_id, fl_link8_id, fr_link7_id, fr_link8_id]:
                    contact_flag = True
                    break
        now_contact_info['finger_contact'] = contact_flag
        now_contact_info_list.append(now_contact_info)
    
    TASK_ENV.history_key_object_contact.append(now_contact_info_list)

    safe_report = {}

    if 'poking' in type_list:
        safe_poking = check_safe_by_poking(TASK_ENV, key_objects, verbose)
        safe_report['poking'] = safe_poking

    if 'topple' in type_list:
        if len(TASK_ENV.history_key_object_pq) > 2:
            safe_topple = check_safe_by_topple(TASK_ENV, key_objects, verbose)
            safe_report['topple'] = safe_topple
        else:
            safe_report['topple'] = True

    if 'fall' in type_list:
        if len(TASK_ENV.history_key_object_pq) > 10:
            safe_fall = check_safe_by_fall(TASK_ENV, key_objects, verbose)
            safe_report['fall'] = safe_fall
        else:
            safe_report['fall'] = True

    return safe_report

def check_safe_by_fall(TASK_ENV, key_objects, verbose):
    safe = True
    for obj_idx, obj in enumerate(key_objects):
        initial_pq = TASK_ENV.history_key_object_pq[0][obj_idx]
        initial_z = initial_pq[2]

        past_in_air = False
        for past_idx in range(-10, -2):
            past_contact_info = TASK_ENV.history_key_object_contact[past_idx][obj_idx]
            past_pq = TASK_ENV.history_key_object_pq[past_idx][obj_idx]
            past_z = past_pq[2]
            if (not past_contact_info['table_contact']) and (not past_contact_info['finger_contact']) and (past_z - initial_z) > 0.06:
                past_in_air = True
        if past_in_air:
            current_contact_info = TASK_ENV.history_key_object_contact[-1][obj_idx]
            if current_contact_info['table_contact']:
                safe = False
                if verbose:
                    print(f"Object {obj_idx} was in air in past 10 frames and now contact table, unsafe!")
            else:
                current_pq = TASK_ENV.history_key_object_pq[-1][obj_idx]
                current_z = current_pq[2]
                table_height = TASK_ENV.table.get_pose().p[2]
                if current_z < table_height - 0.01:
                    safe = False
                    if verbose:
                        print(f"Object {obj_idx} was in air in past 10 frames and now below table height, unsafe!")
    return safe

def check_safe_by_topple(TASK_ENV, key_objects, verbose):
    safe = True
    for obj_idx, obj in enumerate(key_objects):
        all_contacts = TASK_ENV.scene.get_physx_system().get_contacts()
        obj_id = obj.actor.global_id
        contact_to_table = False
        for contact in all_contacts:
            if contact.bodies[0].entity.global_id == obj_id or contact.bodies[1].entity.global_id == obj_id:
                other_body = contact.bodies[0] if contact.bodies[1].entity.global_id == obj_id else contact.bodies[1]
                table_id = TASK_ENV.table.global_id
                if other_body.entity.global_id == table_id:
                    contact_to_table = True
                    break
        if not contact_to_table:
            continue
        
        initial_pq = TASK_ENV.history_key_object_pq[0][obj_idx]
        current_pq = TASK_ENV.history_key_object_pq[-1][obj_idx]

        initial_q_mat = t3d.quaternions.quat2mat(initial_pq[3:7])
        current_q_mat = t3d.quaternions.quat2mat(current_pq[3:7])

        candidate_axes = {0: initial_q_mat[:, 0],
                            1: initial_q_mat[:, 1],
                            2: initial_q_mat[:, 2]}
        max_cos = -1.0
        up_axis_index = -1
        for axis_idx, axis_vec in candidate_axes.items():
            cos_angle = np.dot(axis_vec, np.array([0,0,1])) / (np.linalg.norm(axis_vec) * np.linalg.norm(np.array([0,0,1])))
            if cos_angle > max_cos:
                max_cos = cos_angle
                up_axis_index = axis_idx
        
        initial_up_vector = initial_q_mat[:, up_axis_index]
        current_up_vector = current_q_mat[:, up_axis_index]

        cos_angle = np.dot(initial_up_vector, current_up_vector) / (np.linalg.norm(initial_up_vector) * np.linalg.norm(current_up_vector))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # in radians
        angle_deg = np.degrees(angle)

        if angle_deg > 60.0:  # threshold angle in degrees
            safe = False
            if verbose:
                print(f"Object {obj_idx} topple angle: {angle_deg} degrees and contact to table: {contact_to_table}, unsafe!")
    return safe


def check_safe_by_poking(TASK_ENV, key_objects, verbose=False):
    def get_contacts_for_link(link_id, all_contacts, key_objects_ids):
        return [c for c in all_contacts if (c.bodies[0].entity.global_id == link_id or c.bodies[1].entity.global_id == link_id) and
                (c.bodies[0].entity.global_id in key_objects_ids or c.bodies[1].entity.global_id in key_objects_ids)]
    
    def check_legality_for_contact(contact, finger2world_mat, finger_type, verbose=False):
        contact_point = contact.points[0].position
        contact_normal = contact.points[0].normal
        
        contact_point_hom = np.array([*contact_point, 1.0])
        contact_point_in_finger = np.linalg.inv(finger2world_mat) @ contact_point_hom
        contact_point_in_finger = contact_point_in_finger[:3]
        
        contact_normal_hom = np.array([*contact_normal, 0.0])
        contact_normal_in_finger = np.linalg.inv(finger2world_mat) @ contact_normal_hom
        contact_normal_in_finger = contact_normal_in_finger[:3]

        if 'link7' in finger_type:
            finger_type78 = 'link7'
        elif 'link8' in finger_type:
            finger_type78 = 'link8'
        else:
            raise ValueError("finger_type must contain 'link7' or 'link8'")
        
        legal = judge_point_in_legal_area_legal_normal(contact_point_in_finger, contact_normal_in_finger, finger_type=finger_type78)
        if not legal and verbose:
            print(f'{finger_type} legal: {legal} contact point in finger frame: {contact_point_in_finger}')
        return legal

    links = ['fl_link7', 'fl_link8', 'fr_link7', 'fr_link8']
    link_bodies = {link: TASK_ENV.robot.left_entity.find_link_by_name(link) if 'fl' in link else TASK_ENV.robot.right_entity.find_link_by_name(link) for link in links}
    link_ids = {link: link_bodies[link].entity.global_id for link in links}

    all_contacts = TASK_ENV.scene.get_physx_system().get_contacts()
    key_objects_ids = [obj.actor.global_id for obj in key_objects]

    contact_offset = all_contacts[0].shapes[0].contact_offset if all_contacts else 0
    if contact_offset > 0.00015:
        raise ValueError(f"Contact offset {contact_offset} is too large for poking detection! Must set to 0.0001 or smaller.")


    IGNORE_RIGHT_FINGER = os.environ.get('IGNORE_RIGHT_FINGER', '0')
    if IGNORE_RIGHT_FINGER == '1':
        links = ['fl_link7', 'fl_link8']
    elif IGNORE_RIGHT_FINGER == '0':
        pass
    else:
        raise ValueError("Environment variable 'IGNORE_RIGHT_FINGER' must be str '0' or '1'.")

    legal_all = True
    for link in links:
        link_contacts = get_contacts_for_link(link_ids[link], all_contacts, key_objects_ids)
        if link_contacts:
            finger2world_mat = get_finger2world_matrix(link_bodies[link])
            for contact in link_contacts:
                legal_all = legal_all and check_legality_for_contact(contact, finger2world_mat, link, verbose)

    return legal_all

# ===============================================================

# =================  TOOLS FUNCTIONS  ===========================

# ===============================================================

def get_finger2world_matrix(finger_body):
    pose = finger_body.get_pose()
    p = pose.p
    q = pose.q
    q_mat = t3d.quaternions.quat2mat(q)

    finger2world_mat = np.eye(4)
    finger2world_mat[0:3, 0:3] = q_mat
    finger2world_mat[0:3, 3] = p
    return finger2world_mat

def judge_point_in_legal_area(contact_point_in_finger, finger_type='link7'):
    x, y, z, _ = contact_point_in_finger
    if finger_type == 'link7':
        k = - 0.0245 / 0.071

        flag1 = y < k * x
        flag2 = x < 0.070
        return flag1 and flag2
    elif finger_type == 'link8':
        k = 0.0245 / 0.071
        flag1 = y > k * x
        flag2 = x < 0.070
        return flag1 and flag2
    else:
        raise ValueError("finger_type must be 'link7' or 'link8'")

def judge_point_in_legal_area_legal_normal(contact_point_in_finger, contact_normal_in_finger, finger_type='link7'):
    x, y, z = contact_point_in_finger
    normal_y_thr = 0.9

    if finger_type == 'link7':
        k = - 0.0245 / 0.071
        flag1 = y < k * x
        flag2 = x < 0.070

        area_legal = flag1 and flag2
        if area_legal:
            return True
        else:
            normal_y = contact_normal_in_finger[1]
            if abs(normal_y) < normal_y_thr:
                return False 
            else:
                return True

    elif finger_type == 'link8':
        k = 0.0245 / 0.071
        flag1 = y > k * x
        flag2 = x < 0.070

        area_legal = flag1 and flag2
        if area_legal:
            return True
        else:
            normal_y = contact_normal_in_finger[1]
            if abs(normal_y) < normal_y_thr:
                return False
            else:
                return True

    else:
        raise ValueError("finger_type must be 'link7' or 'link8'")
