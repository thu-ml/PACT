
import torch
import numpy as np
import math

def hat3(omega):  # (3,) -> (3,3)
    w1, w2, w3 = omega
    O = torch.zeros_like(w1)
    return torch.stack([
        torch.stack([O,   -w3,  w2], -1),
        torch.stack([w3,   O,  -w1], -1),
        torch.stack([-w2, w1,   O], -1)
    ], -2)

def hat6(S):  # (6,) -> (4,4)  se(3) hat
    omega = S[:3]
    v = S[3:]
    mat = torch.zeros((4,4), dtype=S.dtype, device=S.device)
    mat[:3, :3] = hat3(omega)
    mat[:3, 3]  = v
    return mat

def exp_twist(S, q):  # S:(6,), q:() -> (4,4)
    # use matrix exponential on se(3)
    return torch.matrix_exp(hat6(S) * q)

def fk_poe(S_list, M, q):
    T = torch.eye(4, dtype=M.dtype, device=M.device)
    for i in range(S_list.shape[0]):
        T = T @ exp_twist(S_list[i], q[i])
    return T @ M


def pose_to_T(pose):
    # pose: SAPIEN Pose, with .to_transformation_matrix() in newer versions
    try:
        return pose.to_transformation_matrix()  # (4,4)
    except AttributeError:
        # fallback: build from pose.p (3,) and pose.q (quat xyzw)
        p = pose.p
        q = pose.q  # [w,x,y,z]
        w,x,y,z = q
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
        ], dtype=np.float64)
        T = np.eye(4, dtype=np.float64)
        T[:3,:3] = R
        T[:3, 3] = p
        return T

def R_of(T): return T[:3,:3]
def p_of(T): return T[:3, 3]


def find_chain(articulation, base_link, ee_link):
    from collections import defaultdict, deque
    graph = defaultdict(list)
    for j in articulation.get_joints():
        graph[j.get_parent_link()].append((j, j.get_child_link()))
    q = deque([(base_link, [])])
    seen = {base_link}
    while q:
        node, path = q.popleft()
        if node == ee_link:
            return [jp[0] for jp in path]
        for j, child in graph[node]:
            if child not in seen:
                seen.add(child)
                q.append((child, path+[(j, child)]))
    raise RuntimeError("EE not reachable")


def build_S_list_and_M_world(articulation, ee_link):
    q_save = articulation.get_qpos().copy()
    articulation.set_qpos(np.zeros_like(q_save))
    if hasattr(articulation, "scene"):
        try: articulation.scene.update_articulation_kinematics()
        except Exception: pass

    links = articulation.get_links()
    base_link = next((L for L in links if L.get_parent() is None), links[0])
    T_wb = pose_to_T(base_link.pose)
    T_we = pose_to_T(ee_link.pose)
    M_world = T_we.copy()

    joint_chain = find_chain(articulation, base_link, ee_link)

    S_list = []
    dof_ids = []
    x_local = np.array([1.0, 0.0, 0.0])



    active_joints = articulation.get_active_joints()

    for j in joint_chain:
        T_wp = pose_to_T(j.get_parent_link().pose)         # world<-parent
        T_pj = pose_to_T(j.get_pose_in_parent())           # parent<-joint
        T_wj = T_wp @ T_pj                                 # world<-joint
        R_wj = R_of(T_wj)
        p_j  = p_of(T_wj)

        axis_world = R_wj @ x_local
        axis_world = axis_world / (np.linalg.norm(axis_world) + 1e-12)

        jtype = j.get_type() if hasattr(j, "get_type") else j.type
        jtype_s = str(jtype).lower()

        if "prismatic" in jtype_s:
            S = np.zeros(6, dtype=np.float64)
            S[3:] = axis_world
        elif "revolute" in jtype_s:
            omega = axis_world
            v = -np.cross(omega, p_j)
            S = np.concatenate([omega, v])
        else:
            continue

        S_list.append(S)

        for _i in range(len(active_joints)):
            if j.name == active_joints[_i].name:
                dof_ids.append(_i)

    articulation.set_qpos(q_save)
    if hasattr(articulation, "scene"):
        try: articulation.scene.update_articulation_kinematics()
        except Exception: pass

    S_np = np.stack(S_list, 0) if S_list else np.zeros((0,6), dtype=np.float64)
    return torch.from_numpy(S_np).double(), torch.from_numpy(M_world).double(), dof_ids


def get_gripper_keypoints(S, M, q, bias):
    joint = q[:,:6]
    tips = q[:,6:]
    gripper_positions = {
        "forward":[],
    }
    gripper_directions = {
        "forward":[],
        "tips1":[],
        "tips2":[],
        "top":[]
    }

    for q_i in joint:
        T_ee = fk_poe(S, M, q_i)
        gripper_positions["forward"].append(T_ee[:3, 3])
        gripper_directions["forward"].append(T_ee[:3, 0]) # x
        gripper_directions["tips1"].append(T_ee[:3, 1]) # y
        gripper_directions["tips2"].append(-T_ee[:3, 1]) # y
        gripper_directions["top"].append(T_ee[:3, 2]) # z

    for k in gripper_directions.keys():
        ds = torch.stack(gripper_directions[k], dim=0)
        ds_norm = ds / torch.norm(ds, dim=1, keepdim=True)
        gripper_directions[k] = ds_norm
    
    gripper_positions["forward"] = torch.stack(gripper_positions["forward"], dim=0)
    gripper_positions["forward"] = gripper_positions["forward"] + gripper_directions["forward"] * bias
    gripper_positions["tips1"] = gripper_positions["forward"] + gripper_directions["tips2"] * tips * 0.04
    gripper_positions["tips2"] = gripper_positions["forward"] + gripper_directions["tips1"] * tips * 0.04

    return gripper_positions, gripper_directions
    


def local_keypoint_to_world(local_position, local_direction, T_ee):
    local_pos_homo = torch.cat([local_position, torch.tensor([1.0], device=T_ee.device)])
    
    world_pos_homo = T_ee @ local_pos_homo
    world_position = world_pos_homo[:3]
    
    R_ee = T_ee[:3, :3]
    world_direction = R_ee @ local_direction

    world_direction = world_direction / torch.norm(world_direction)
    
    return world_position, world_direction

def get_gripper_keypoints_ex(S, M, q, local_position, local_direction):
    joint = q[:6]
    T_ee = fk_poe(S, M, joint) 

    return local_keypoint_to_world(local_position, local_direction, T_ee)



def world_keypoint_to_local(world_position, world_direction, T_ee):
    T_ee_inv = torch.inverse(T_ee)

    world_pos_homo = torch.cat([world_position, torch.tensor([1.0], device=T_ee.device)])
    
    local_pos_homo = T_ee_inv @ world_pos_homo
    local_position = local_pos_homo[:3]
    
    R_ee_inv = T_ee_inv[:3, :3]
    local_direction = R_ee_inv @ world_direction
    
    local_direction = local_direction / torch.norm(local_direction)
    
    return local_position, local_direction

def capture_keypoint_in_local_frame(S_list, M, q, world_position, world_direction):
    T_ee = fk_poe(S_list, M, q)
    
    local_position, local_direction = world_keypoint_to_local(
        torch.tensor(world_position, dtype=T_ee.dtype, device=T_ee.device),
        torch.tensor(world_direction, dtype=T_ee.dtype, device=T_ee.device),
        T_ee
    )
    
    return local_position, local_direction