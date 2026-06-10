from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *

class handover_apple(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

        self.chosen_contact_point_id = None

    def load_actors(self):

        self.apple_id = 1
        self.apple = rand_create_actor(
            self,
            xlim=[-0.25, -0.05],
            ylim=[0.03, 0.23],
            modelname="035_apple",
            rotate_rand=True,
            rotate_lim=[0, 1, 0],
            qpos=[0.66, 0.66, -0.25, -0.25],
            convex=True,
            model_id=self.apple_id,
        )

        self.add_prohibit_area(self.apple, padding=0.1)

    def play_once(self):
        grasp_arm_tag = ArmTag("left" if self.apple.get_pose().p[0] < 0 else "right")
        place_arm_tag = grasp_arm_tag.opposite

        self.move(
            self.grasp_actor(
                self.apple,
                arm_tag=grasp_arm_tag,
                pre_grasp_dis=0.07,
                grasp_dis=0.0,
            ))

        self.move(self.move_by_displacement(grasp_arm_tag, y=-0.1, z=0.1))
        target_pose = np.zeros(7)
        target_pose[:3] = [-self.robot.left_gripper_bias, 0, 0.9]
        target_pose_q_mat = np.array([[1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1]])
        target_pose[3:] = t3d.quaternions.mat2quat(target_pose_q_mat)

        self.move(self.move_to_pose(arm_tag=grasp_arm_tag, target_pose=target_pose))


        if  self.chosen_contact_point_id is not None:
            used_contact_point = self.apple.get_contact_point(self.chosen_contact_point_id)
            used_contact_point_q = used_contact_point[3:]
            used_contact_point_q_mat = t3d.quaternions.quat2mat(used_contact_point_q)
            used_contact_point_0_direction = used_contact_point_q_mat[:3, 0]
            used_contact_point_1_direction = used_contact_point_q_mat[:3, 1]
            used_contact_point_2_direction = used_contact_point_q_mat[:3, 2]

            candidate_contact_point_id_list = []
            for i in range(100):
                p = self.apple.get_contact_point(i)
                if p is None:
                    break
                p_q = p[3:]
                p_q_mat = t3d.quaternions.quat2mat(p_q)
                p_1_direction = p_q_mat[:3, 1]
                cos_angle = np.dot(used_contact_point_1_direction, p_1_direction) / (
                            np.linalg.norm(used_contact_point_1_direction) * np.linalg.norm(p_1_direction))
                if cos_angle < -0.7:
                    p_2_direction = p_q_mat[:3, 2]
                    cos_angle_up = np.dot(used_contact_point_2_direction, p_2_direction) / (
                                np.linalg.norm(used_contact_point_2_direction) * np.linalg.norm(p_2_direction))
                    if abs(cos_angle_up) < 0.3:
                        candidate_contact_point_id_list.append(i)

            for i in candidate_contact_point_id_list:
                p = self.apple.get_contact_point(i)
                p_q = p[3:]
                p_q_mat = t3d.quaternions.quat2mat(p_q)
                p_1_direction = p_q_mat[:3, 1]
        
        else:
            candidate_contact_point_id_list = None



        self.move(
            self.grasp_actor(
                self.apple,
                arm_tag=place_arm_tag,
                pre_grasp_dis=0.07,
                grasp_dis=0.0,
                contact_point_id=candidate_contact_point_id_list
            ))
        
        self.move(self.open_gripper(grasp_arm_tag))
        self.move(self.move_by_displacement(grasp_arm_tag, x=-0.1, move_axis="world"),
                  self.move_by_displacement(place_arm_tag, x=0.1, move_axis="world")
                  )

        a = self.check_success()

        return self.info


    
    def get_left_gripper_tip(self):
        left_ee_pose = self.robot.left_ee.global_pose
        left_ee_p = left_ee_pose.p
        left_ee_q = left_ee_pose.q
        left_ee_q_mat = t3d.quaternions.quat2mat(left_ee_q)
        left_ee_bias = self.robot.left_gripper_bias - 0.02
        left_ee_forward_direction = left_ee_q_mat[:3, 0]
        left_ee_forward_direction_norm = left_ee_forward_direction / np.linalg.norm(left_ee_forward_direction)
        gripper_tip_xyz_left = left_ee_p + left_ee_forward_direction_norm * left_ee_bias
        return gripper_tip_xyz_left

    def get_right_gripper_tip(self):
        right_ee_pose = self.robot.right_ee.global_pose
        right_ee_p = right_ee_pose.p
        right_ee_q = right_ee_pose.q
        right_ee_q_mat = t3d.quaternions.quat2mat(right_ee_q)
        right_ee_bias = self.robot.right_gripper_bias - 0.02
        right_ee_forward_direction = right_ee_q_mat[:3, 0]
        right_ee_forward_direction_norm = right_ee_forward_direction / np.linalg.norm(right_ee_forward_direction)
        gripper_tip_xyz_right = right_ee_p + right_ee_forward_direction_norm * right_ee_bias
        return gripper_tip_xyz_right


    def check_success(self):
        apple_pos = self.apple.get_pose().p

        condition_left_open = self.is_left_gripper_open()

        left_gripper_tip_xyz = self.get_left_gripper_tip()
        condition_left_distance = np.linalg.norm(left_gripper_tip_xyz - apple_pos) > 0.1

        condition_right_close = self.is_right_gripper_close()

        right_gripper_tip_xyz = self.get_right_gripper_tip()
        condition_right_distance = np.linalg.norm(right_gripper_tip_xyz - apple_pos) < 0.05

        return condition_left_open and condition_left_distance and condition_right_close and condition_right_distance
