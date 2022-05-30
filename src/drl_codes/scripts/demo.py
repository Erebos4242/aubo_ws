#!/usr/bin/env python3
import sys
import copy

from scipy import rand
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from time import time
import numpy as np
from ctypes import *
import ctypes
from std_msgs.msg import *
from random import random
from sensor_msgs.msg import Image
import numpy as np
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import Header
from aubo_msgs.msg import GraspPosition
from image_frame_to_world_frame import image_frame_to_world_frame
from time import time
import threading
from scipy.spatial.transform import Rotation


def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    all_equal = True
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

    return True


class RobotSim(object):
    def __init__(self):
        super(RobotSim, self).__init__()

        """move group"""
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('RobotSim', anonymous=True)
        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        group_name = "robot"
        move_group = moveit_commander.MoveGroupCommander(group_name)
        planning_frame = move_group.get_planning_frame()
        eef_link = move_group.get_end_effector_link()
        group_names = robot.get_group_names()

        bridge = CvBridge()

        """move group"""
        self.box_name = 'table_surface'
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names

        """global variable"""
        self.init_joint = [0, 0, 0, 0, 0, 0]
        self.add_box()
        self.robot_ready = True
        self.robot_init = False
        self.graper_ready = False
        self.table_height = 1.5342424
        self.image_height = 1080
        self.image_width = 1920
        self.bridge = bridge
        self.x_per_pixel = 0.5 / 541
        self.y_per_pixel = 0.5 / 541
        self.camera_height = 2.5
        self.lock = threading.RLock()

        """publisher"""
        display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                       moveit_msgs.msg.DisplayTrajectory,
                                                       queue_size=20)
        grasp_position_map_pub = rospy.Publisher('/grasp/map', Image, queue_size=10)
        grasp_position_pub = rospy.Publisher('/grasp/position', GraspPosition, queue_size=1)
        graper_command = rospy.Publisher('/aubo_i5/graper_position_controller/command', Float64, queue_size=10)
        grasp_target = rospy.Publisher('/grasp/target', GraspPosition, queue_size=1)

        """Subscriber"""
        rospy.Subscriber("/camera/depth/image_raw", Image, self.get_position)
        rospy.Subscriber("/grasp/position", GraspPosition, self.grasp_strategy)
        rospy.Subscriber('/grasp/target', GraspPosition, self.grasp_excution)

        """publisher"""
        self.display_trajectory_publisher = display_trajectory_publisher
        self.graper_command = graper_command
        self.grasp_position_map_pub = grasp_position_map_pub
        self.grasp_position_pub = grasp_position_pub
        self.grasp_target = grasp_target

        """init env"""
        self.init_joint_state()
        self.add_box()

    """Perception"""

    def get_position(self, img):
        new_img = self.bridge.imgmsg_to_cv2(img, "passthrough")
        depth_array = np.array(new_img, dtype=np.float32)
        depth = depth_array.copy()
        cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
        image = np.stack((depth_array * 255,) * 3, axis=-1)
        image = image.astype(np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.robot_init:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

            object_position = []  # x, y, z, rotation, width
            for index in range(len(contours)):
                rect = cv2.minAreaRect(contours[index])
                center_x, center_y = int(rect[0][1]), int(rect[0][0])
                h = depth[center_x][center_y]

                # if rect[1][0] >= rect[1][1]:
                #     rotation = (rect[2])
                #     target_width = rect[1][1]
                # else:
                #     rotation = (90 + rect[2])
                #     target_width = rect[1][0]

                if rect[1][0] >= rect[1][1]:
                    rotation = -rect[2]
                    target_width = rect[1][1]
                else:
                    rotation = (90 - rect[2])
                    target_width = rect[1][0]

                res = list(self.image_frame_to_world_frame(h, target_width, center_x, center_y))

                if (self.table_height - h) / 2 > 0.04:
                    res[2] = self.camera_height - h - 0.04
                else:
                    res[2] = self.camera_height - self.table_height + (self.table_height - h) / 2
                res.append(rotation)
                object_position += list(res)
                cv2.putText(image, "(%.3f, %.3f, %.3f, %.3f)" % (res[0], res[1], res[2], res[4]),
                            (center_y - 220, center_x), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 1, cv2.LINE_AA)

            self.grasp_position_pub.publish(object_position)

        image_temp = Image()
        header = Header(stamp=rospy.Time.now())
        header.frame_id = 'map'
        image_temp.height = self.image_height
        image_temp.width = self.image_width
        image_temp.encoding = 'rgb8'
        image_temp.data = np.array(image).tostring()
        image_temp.header = header
        image_temp.step = self.image_width * 3
        self.grasp_position_map_pub.publish(image_temp)

    """Decision"""

    def grasp_strategy(self, data):
        if not self.robot_init:
            return
        index = 0
        grasp_position = data.grasp_position[index: index + 5]
        self.grasp_target.publish(grasp_position)

    """Control"""

    def init_joint_state(self):
        move_group = self.move_group
        joint_goal = move_group.get_current_joint_values()
        joint_goal[0] = self.init_joint[0]
        joint_goal[1] = self.init_joint[1]
        joint_goal[2] = self.init_joint[2]
        joint_goal[3] = self.init_joint[3]
        joint_goal[4] = self.init_joint[4]
        joint_goal[5] = self.init_joint[5]
        current_joints = move_group.get_current_joint_values()
        # if all_close(joint_goal, current_joints, 0.01):
        #     print(time(), "already init")
        #     return
        print(time(), "init joint====================================")
        move_group.go(joint_goal, wait=True)
        move_group.stop()
        self.grasp(0)
        print(time(), "init joint done")
        self.robot_init = True

    def go_to_pose_goal(self, x, y, z, r):
        self.robot_init = False
        print(time(), "go to pose goal(x, y, z, r)", x, y, z, r)
        move_group = self.move_group
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.position.x = x
        pose_goal.position.y = y
        pose_goal.position.z = z

        quat = self.matrix_to_quaternion(-r)

        pose_goal.orientation.x = quat[0]
        pose_goal.orientation.y = quat[1]
        pose_goal.orientation.z = quat[2]
        pose_goal.orientation.w = quat[3]

        current_pose = self.move_group.get_current_pose().pose
        # if all_close(pose_goal, current_pose, 0.01):
        #     print(time(), "already at pose goal")
        #     return
        move_group.set_pose_target(pose_goal)
        plan = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        print(time(), "go to pose goal, done")

    def place(self):
        x = -0.2 + 0.3 * random()
        y = -0.3 + 0.6 * random()
        z = 1.3
        r = 180 * random()
        self.go_to_pose_goal(x, y, z, r)
        self.grasp(0)

    def grasp(self, data):
        data = 0.9 / 0.1 * data
        position = Float64()
        position.data = data
        rate = rospy.Rate(10)  # 10hz
        start = time()
        while not rospy.is_shutdown():
            self.graper_command.publish(data)
            rate.sleep()
            end = time()
            if end - start > 2:
                break

    def grasp_excution(self, data):
        if not self.robot_init:
            return
        target = data.grasp_position
        print(time(), "grasp_excution, go to")
        self.go_to_pose_goal(target[0], target[1], target[2] + 0.15, target[4])
        print(time(), "grasp_excution, grasp")
        self.grasp(target[3])
        print(time(), "grasp_excution, place")
        self.place()
        print(time(), "grasp_excution, init")
        self.init_joint_state()

    """Robot Environment"""

    def add_box(self, timeout=4):
        box_name = self.box_name
        scene = self.scene

        rospy.sleep(1)
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = self.move_group.get_planning_frame()
        box_pose.pose.orientation.w = 0
        box_pose.pose.orientation.x = 1.0
        box_pose.pose.orientation.y = 0
        box_pose.pose.orientation.z = 0
        box_pose.pose.position.z = 0.97
        box_name = "table"
        scene.add_box("table", box_pose, size=(2, 2, 0.01))

    """Convenience method"""

    # def matrix_to_quaternion(self, r):
    #     tran_func = cdll.LoadLibrary('libmatrix_to_quaternion.so')
    #     tran_func.rotationMatrix2Quaterniond.argtypes = [c_double]
    #     tran_func.rotationMatrix2Quaterniond.restype = ctypes.POINTER(ctypes.c_double * 4)

    #     res = tran_func.rotationMatrix2Quaterniond(r)
    #     return res.contents[0], res.contents[1], res.contents[2], res.contents[3]

    def matrix_to_quaternion(self, r):
        r = -(r / 180) * np.pi
        M1 = np.array([[1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]])
        M2 = np.array([[np.cos(r), np.sin(r), 0],
                    [-np.sin(r), np.cos(r), 0],
                    [0, 0, 1]])
        M1 = np.mat(M1)
        M2 = np.mat(M2)
        M = M1 * M2

        return Rotation.from_matrix(M).as_quat()

    def w_h_per_pixel(self, h):
        tan_w = self.x_per_pixel * self.image_width / 2 / self.table_height
        tan_h = self.y_per_pixel * self.image_height / 2 / self.table_height
        w_per_pixel = (tan_w * h * 2) / 1920
        h_per_pixel = (tan_h * h * 2) / 1080
        return w_per_pixel, h_per_pixel

    def image_frame_to_world_frame(self, h, w, u, v):
        w_per_pixel, h_per_pixel = self.w_h_per_pixel(h)
        x = (self.image_width / 2 - v) * w_per_pixel
        y = (self.image_height / 2 - u) * h_per_pixel
        return y, x, self.camera_height - h, w * (w_per_pixel + h_per_pixel) / 2


def main():
    sim = RobotSim()
    # sim.go_to_pose_goal(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
    # while True:
    #     print("send=======================================")
    #     sim.place()
    # # rospy.sleep(5)
    # sim.go_to_joint_state()
    rospy.spin()


if __name__ == "__main__":
    main()
