import torch

# def objects_collision_cost(keypoint1_pos, keypoint2_pos):
#     cost = -torch.norm(keypoint1_pos-keypoint2_pos, p=2)
    
#     return cost

def behavior_alignment_cost(object2_vector, object1_keypoint_pos, object2_keypoint_pos, keypoints_offset, lambda_param):
    lA = object1_keypoint_pos - object2_keypoint_pos

    I = torch.eye(3, device=object2_vector.device, dtype=object2_vector.dtype)
    zzT = torch.outer(object2_vector, object2_vector)
    projection_matrix = I - zzT

    lateral_component = projection_matrix @ lA
    cost_1 = torch.norm(lateral_component, p=2) ** 2

    vertical_component = torch.dot(object2_vector, lA)
    cost_2 = (vertical_component - keypoints_offset) ** 2
    
    cost = cost_1 + lambda_param * cost_2
    return cost

def grippers_poking_cost(keypoint_pos, gripper_pos, approach_vector):
    keypoint_to_tip = keypoint_pos - gripper_pos

    I = torch.eye(3, device=approach_vector.device, dtype=approach_vector.dtype)
    aaT = torch.outer(approach_vector, approach_vector)
    projection_matrix = I - aaT

    lateral_component = projection_matrix @ keypoint_to_tip
    cost = torch.norm(lateral_component, p=2)
    
    return cost

# def grippers_tearing_cost(left_gripper_pos, right_gripper_pos, initial_grasp_width):
#     cost = (torch.norm(left_gripper_pos-right_gripper_pos, p=2) - initial_grasp_width)**2
    
#     return cost

# def grippers_collision_cost(left_gripper_pos, right_gripper_pos):
#     cost = -torch.norm(left_gripper_pos-right_gripper_pos, p=2)
    
#     return cost

def grippers_rotation_cost(rotate_vector, contact_vectors): 
    rotate_vector = torch.nn.functional.normalize(rotate_vector, dim=0)
    contact_vectors = torch.nn.functional.normalize(contact_vectors, dim=1)
    cosine_similarities = torch.matmul(contact_vectors, rotate_vector)
    closest_similarity, closest_idx = torch.max(cosine_similarities, dim=0)
    cost = 1.0 - closest_similarity
    return cost

def grippers_distance_cost(keypoint_pos, gripper_pos):
    cost = torch.sum((keypoint_pos - gripper_pos)**2)

    return cost

def quaternion_to_direction(q):
    q = q / torch.linalg.norm(q)
    
    def quat_mult(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return torch.tensor([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], device=q.device, dtype=torch.float64)
    
    v = torch.tensor([0., 0., 0., 1.], device=q.device, dtype=q.dtype)
    q_conj = torch.tensor([q[0], -q[1], -q[2], -q[3]])
    v_rot = quat_mult(quat_mult(q, v), q_conj)
    direction = v_rot[1:] / torch.linalg.norm(v_rot[1:])
    return direction

def build_cube_side_normals(given_side_normal, top_normal=torch.tensor([0., 0., 1.], dtype=torch.float64)):
    given_side_normal = torch.nn.functional.normalize(given_side_normal, dim=0)
    top_normal = torch.nn.functional.normalize(top_normal, dim=0).to(given_side_normal.device)
    
    side2 = torch.cross(given_side_normal, top_normal)
    side2 = torch.nn.functional.normalize(side2, dim=0)
    side3 = -given_side_normal
    side4 = -side2
    side_normals = torch.stack([given_side_normal, side2, side3, side4])
    
    return side_normals

