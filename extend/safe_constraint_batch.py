import torch

def behavior_alignment_cost_batch(
    object2_vectors: torch.Tensor,
    object1_keypoint_pos: torch.Tensor,
    object2_keypoint_pos: torch.Tensor,
    keypoints_offset: float,
    lambda_param: float
):
    """
    Args:
        object2_vectors:        [3]
        object1_keypoint_pos:   [N, 3]
        object2_keypoint_pos:   [3]
        keypoints_offset:       float
        lambda_param:           float

    Returns:
        cost:                   [N]
    """
    N = object1_keypoint_pos.shape[0]
    object2_vectors = object2_vectors.repeat(N, 1)           # N -> [N,3]    
    object2_keypoint_pos = object2_keypoint_pos.repeat(N, 1) # N -> [N,3]
    # lA = p1 - p2
    lA = object1_keypoint_pos - object2_keypoint_pos         # [N,3]

    z = torch.nn.functional.normalize(object2_vectors, dim=1)# [N,3]

    # projection matrix: I - zz^T
    zzT = z[:, :, None] * z[:, None, :]                      # [N,3,3]
    I = torch.eye(3, device=z.device, dtype=z.dtype).expand_as(zzT)
    P = I - zzT                                              # [N,3,3]

    # lateral component
    lateral = torch.bmm(P, lA.unsqueeze(-1)).squeeze(-1)     # [N,3]
    cost_1 = torch.sum(lateral ** 2, dim=1)                  # [N]

    # vertical component
    vertical = torch.sum(z * lA, dim=1)                      # [N]
    cost_2 = (vertical - keypoints_offset) ** 2              # [N]

    cost = cost_1 + lambda_param * cost_2
    return cost

def grippers_poking_cost_batch(
    keypoint_pos: torch.Tensor,
    gripper_pos: torch.Tensor,
    approach_vector: torch.Tensor
):
    """
    keypoint_pos:    [3]
    gripper_pos:     [N,3]
    approach_vector: [N,3]
    return:          [N]
    """
    keypoint_to_tip = keypoint_pos[None, :] - gripper_pos  # [N,3]

    aaT = approach_vector[:, :, None] * approach_vector[:, None, :]  # [N,3,3]
    I = torch.eye(3, device=gripper_pos.device).expand_as(aaT)

    proj = I - aaT
    lateral = torch.bmm(proj, keypoint_to_tip.unsqueeze(-1)).squeeze(-1)

    return torch.norm(lateral, dim=1)

def grippers_distance_cost_batch(
    keypoint_pos: torch.Tensor,
    gripper_pos: torch.Tensor
):
    """
    keypoint_pos: [3]
    gripper_pos:  [N,3]
    return:       [N]
    """
    return torch.sum((keypoint_pos[None, :] - gripper_pos) ** 2, dim=1)

def grippers_rotation_cost_batch(
    rotate_vectors: torch.Tensor,
    contact_vectors: torch.Tensor
):
    """
    rotate_vectors:  [N, 3]    (gripper top direction, possibly detached)
    contact_vectors: [K, 3]    (block contact directions, fixed)
    return:          [N]
    """
    # normalize
    rotate_vectors = torch.nn.functional.normalize(rotate_vectors, dim=1)   # [N,3]
    contact_vectors = torch.nn.functional.normalize(contact_vectors, dim=1) # [K,3]

    # cosine similarity: [N,K]
    cosine = torch.matmul(rotate_vectors, contact_vectors.T)

    # best alignment per timestep
    closest_sim, _ = torch.max(cosine, dim=1)  # [N]

    return 1.0 - closest_sim