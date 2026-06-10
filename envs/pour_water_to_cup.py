from ._base_task import Base_Task
from .utils import *


class pour_water_to_cup(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.bottle_id = 16
        self.cup_id = 2

        self.cup = rand_create_actor(
            self,
            xlim=[0.05, 0.25],
            ylim=[-0.03, 0.17],
            modelname="021_cup",
            rotate_rand=True,
            rotate_lim=[0, 1, 0],
            qpos=[0.27, 0.27, -0.65, -0.65],
            convex=True,

            model_id=self.cup_id,
        )
        # cup_pose = self.cup.get_pose().p

        self.bottle = rand_create_actor(
            self,
            xlim=[-0.25, -0.15],
            ylim=[-0.03, 0.17],
            modelname="001_bottle",
            rotate_rand=True,
            rotate_lim=[0, 1, 0],
            qpos=[0.65, 0.65, 0.27, 0.27],
            convex=True,
            model_id=self.bottle_id,
        )

        self.add_prohibit_area(self.cup, padding=0.05)
        self.add_prohibit_area(self.bottle, padding=0.05)
        self.delay(2)

    def play_once(self):
        self.move(
            self.grasp_actor(self.bottle, arm_tag="left", pre_grasp_dis=0.1, gripper_pos=0),
            self.grasp_actor(self.cup, arm_tag="right", pre_grasp_dis=0.1, gripper_pos=0)
            )
        
        bottle_target_pose_np = self.bottle.get_functional_point(0)
        bottle_target_pose_np[:3] = [-0.1, 0, bottle_target_pose_np[2]+0.1]
        cup_target_pose_np = self.cup.get_functional_point(0)
        cup_target_pose_np[:3] = [0.1, 0, cup_target_pose_np[2]+0.1]
        self.move(
            self.place_actor(
                self.bottle,
                target_pose=bottle_target_pose_np,
                arm_tag="left",
                functional_point_id=0,
                pre_dis=0.0,
                dis=0,
                is_open=False,
            ),
            self.place_actor(
                self.cup,
                target_pose=cup_target_pose_np,
                arm_tag="right",
                functional_point_id=0,
                pre_dis=0.0,
                dis=0,
                is_open=False,
            )
        )

        axis1 = [0, -1, 0]
        axis2 = [0, 0, 1]
        axis3 = [-1, 0, 0]
        q_cup_mat_target = np.array([
            axis1,
            axis2,
            axis3,
        ]).T 
        q_cup_target = t3d.quaternions.mat2quat(q_cup_mat_target)
        target_pose_np = self.cup.get_functional_point(1)
        target_pose_np[0] -= 0.12
        target_pose_np[2] += 0.05
        target_pose_np[3:] = q_cup_target
        bottle_functional_point_index = 0

        # bottle_functional_point_pos = self.bottle.get_functional_point(bottle_functional_point_index)
        bottle_functional_point_q = self.bottle.get_functional_point(bottle_functional_point_index)[3:]

        world_rot_quat = t3d.quaternions.mat2quat(t3d.euler.euler2mat(0, np.pi/2, 0))

        bottle_functional_point_q_rotated = t3d.quaternions.qmult(world_rot_quat, bottle_functional_point_q)

        target_pose_np[3:] = bottle_functional_point_q_rotated

        self.move(
            self.place_actor(
                self.bottle,
                target_pose=target_pose_np,
                arm_tag="left",
                functional_point_id=bottle_functional_point_index,
                pre_dis=0.0,
                pre_dis_axis='fp',
                dis=0,
                is_open=False,
            ))
        
        self.delay(1)

        self.info["info"] = {
            "{A}": f"001_bottle/base{self.bottle_id}",
            "{B}": f"021_cup/base{self.cup_id}",
        }
        
        return self.info
        


    def check_success(self):
        cup_up_direction = t3d.quaternions.quat2mat(self.cup.get_pose().q)[:3,1]
        condition_cup_upright = cup_up_direction[2] > 0.9

        bottle_up_direction = t3d.quaternions.quat2mat(self.bottle.get_pose().q)[:3,1]
        bottle_up_angle_with_xoy = math.acos(abs(np.clip(bottle_up_direction[2], -1.0, 1.0)))
        condition_bottle_tilted = bottle_up_angle_with_xoy > math.pi / 180 * 60

        bottle_up_direction = t3d.quaternions.quat2mat(self.bottle.get_pose().q)[:3,1]
        bottle_bottom_pos = self.bottle.get_pose().p
        bottle_height = self.bottle.config['extents'][1] * self.bottle.config['scale'][1]
        bottle_top_pos = bottle_bottom_pos + bottle_up_direction * bottle_height

        cup_up_direction = t3d.quaternions.quat2mat(self.cup.get_pose().q)[:3,1]
        cup_bottom_pos = self.cup.get_pose().p
        cup_height = self.cup.config['extents'][1] * self.cup.config['scale'][1]

        cup_mouth_diameter = min(self.cup.config['extents'][0] * self.cup.config['scale'][0], 
                                 self.cup.config['extents'][2] * self.cup.config['scale'][2]) * 1.2

        cup_top_pos = cup_bottom_pos + cup_up_direction * cup_height
        condition_bottle_above_cup = bottle_top_pos[2] > cup_top_pos[2] and np.linalg.norm(bottle_top_pos[:2] - cup_top_pos[:2]) < cup_mouth_diameter/2

        cup_bottom_pos = self.cup.get_pose().p 
        cup_higher_than_table = cup_bottom_pos[2] > 0.74 + 0.02
        bottle_bottom_pos = self.bottle.get_pose().p
        bottle_higher_than_table = bottle_bottom_pos[2] > 0.74 + 0.02
        condition_cup_bottle_lifted = cup_higher_than_table and bottle_higher_than_table

        return condition_cup_upright and condition_bottle_tilted and condition_bottle_above_cup and condition_cup_bottle_lifted



import math

def tilt_quaternion_xy(vx, vy, theta):
    n = math.hypot(vx, vy)
    if n < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)

    dx, dy = vx / n, vy / n
    ax, ay, az = -dy, dx, 0.0

    half = 0.5 * theta
    s = math.sin(half)
    w = math.cos(half)
    x = ax * s
    y = ay * s
    z = az * s
    return (w, x, y, z)





import numpy as np
def save_point_cloud_color(xyz: np.ndarray, color: np.ndarray, filename: str = "point_cloud.ply"):
    """
    Save a point cloud with color to a PLY file.

    Args:
        xyz (np.ndarray): Nx3 array of XYZ coordinates.
        color (np.ndarray): Nx3 array of RGB values (0-255).
        filename (str): Output filename.
    """
    assert xyz.shape[0] == color.shape[0], "xyz and color must have same number of points"
    assert xyz.shape[1] == 3 and color.shape[1] == 3, "xyz and color must be Nx3 arrays"

    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for pt, col in zip(xyz, color):
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {int(col[0])} {int(col[1])} {int(col[2])}\n")