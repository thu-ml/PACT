
import torch
import numpy as np
import math

def hat3_batch(omega: torch.Tensor):
    """
    omega: [N,3]
    return: [N,3,3]
    """
    w1, w2, w3 = omega[:, 0], omega[:, 1], omega[:, 2]
    O = torch.zeros_like(w1)
    return torch.stack([
        torch.stack([O,   -w3,  w2], -1),
        torch.stack([w3,   O,  -w1], -1),
        torch.stack([-w2, w1,   O], -1)
    ], -2)

def hat6_batch(S: torch.Tensor):
    """
    S: [N,6]
    return: [N,4,4]
    """
    omega = S[:, :3]
    v     = S[:, 3:]

    mat = torch.zeros((S.shape[0], 4, 4), dtype=S.dtype, device=S.device)
    mat[:, :3, :3] = hat3_batch(omega)
    mat[:, :3,  3] = v
    return mat

def exp_twist_batch(S: torch.Tensor, q: torch.Tensor):
    """
    S: [N,6]
    q: [N]
    return: [N,4,4]
    """
    return torch.matrix_exp(hat6_batch(S) * q[:, None, None])

def fk_poe_batch(S_list: torch.Tensor, M: torch.Tensor, q: torch.Tensor):
    """
    S_list: [J,6]
    q:      [N,J]
    M:      [4,4]
    return: [N,4,4]
    """
    N, J = q.shape
    T = torch.eye(4, device=M.device, dtype=M.dtype).expand(N,4,4).clone()

    for j in range(J):
        T = T @ exp_twist_batch(S_list[j].expand(N,6), q[:, j])

    return T @ M
  
def get_gripper_keypoints_batch(S: torch.Tensor, M: torch.Tensor, q: torch.Tensor, bias: float):
    """
    q: [N,7]  → 6 joints + 1 gripper
    return:
      positions["forward"]: [N,3]
      directions[k]:        [N,3]
    """
    joint = q[:, :6]  # [N,6]
    tips  = q[:, 6:]  # [N,1]

    T_ee = fk_poe_batch(S, M, joint)  # [N,4,4]

    forward = T_ee[:, :3, 3]
    x_axis  = T_ee[:, :3, 0]
    y_axis  = T_ee[:, :3, 1]
    z_axis  = T_ee[:, :3, 2]

    def norm(v): return v / torch.norm(v, dim=1, keepdim=True)

    directions = {
        "forward": norm(x_axis),
        "tips1":   norm(y_axis),
        "tips2":   norm(-y_axis),
        "top":     norm(z_axis),
    }

    positions = {}
    positions["forward"] = forward + directions["forward"] * bias
    positions["tips1"] = positions["forward"] + directions["tips2"] * tips * 0.04
    positions["tips2"] = positions["forward"] + directions["tips1"] * tips * 0.04

    return positions, directions

