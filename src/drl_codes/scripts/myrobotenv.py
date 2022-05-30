#!/usr/bin/env python3
from multiprocessing.connection import wait
from os import sep
from re import T
import sys
import copy
from turtle import st

from scipy import rand
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs
from geometry_msgs.msg import Pose
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
from gazebo_msgs.msg import ModelStates, ModelState
from std_srvs.srv import Empty
from gazebo_msgs.srv import DeleteModel, SetModelState, SpawnModel
from tf.transformations import quaternion_from_euler, euler_from_quaternion


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
        self.arm = move_group
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names

        """global variable"""
        self.init_joint = [1.571, 0, 0, 0, 0, 0]
        self.scene_models = ['ground_plane', 'table_marble', 'kinect', 'background', 'aubo_i5']
        self.object_models = ['cleaner', 'eraser', 'shampoo', 'chewinggum', 'salt']
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
        self.reward = 0

        """publisher"""
        display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                       moveit_msgs.msg.DisplayTrajectory,
                                                       queue_size=20)
        grasp_position_map_pub = rospy.Publisher('/grasp/map', Image, queue_size=10)
        grasp_position_pub = rospy.Publisher('/grasp/position', GraspPosition, queue_size=1)
        graper_command = rospy.Publisher('/aubo_i5/graper_position_controller/command', Float64, queue_size=10)
        grasp_target = rospy.Publisher('/grasp/target', GraspPosition, queue_size=1)

        # """Subscriber"""
        # rospy.Subscriber("/camera/depth/image_raw", Image, self.get_position)
        # rospy.Subscriber("/grasp/position", GraspPosition, self.grasp_strategy)
        # rospy.Subscriber('/grasp/target', GraspPosition, self.grasp_excution)

        """publisher"""
        self.display_trajectory_publisher = display_trajectory_publisher
        self.graper_command = graper_command
        self.grasp_position_map_pub = grasp_position_map_pub
        self.grasp_position_pub = grasp_position_pub
        self.grasp_target = grasp_target

        """service"""
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        self.spawn = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)

        """init env"""
        self.init_joint_state()
        self.add_box()

    ######################################################################
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

    def get_img(self):
        # Get color image from simulation
        # rospy.sleep(1)
        raw_img = rospy.wait_for_message('/camera/color/image_raw', Image)
        raw_image = CvBridge().imgmsg_to_cv2(raw_img, "rgb8")

        color_img = np.asarray(raw_image)
        color_img = color_img.astype(float) / 255
        color_img[color_img < 0] += 1
        color_img *= 255
        color_img = color_img.astype(np.uint8)

        # Get depth image from simulation
        depth_img = rospy.wait_for_message('/camera/depth/image_raw', Image)
        rospy.sleep(0.01)
        new_img = CvBridge().imgmsg_to_cv2(depth_img, "passthrough")
        depth_array = np.array(new_img, dtype=np.float32)
        cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
        image = np.stack((depth_array * 255,) * 3, axis=-1)
        image = image.astype(np.uint8)

        depth_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return color_img, depth_img
    
    def get_model_states(self):
        states = rospy.wait_for_message('/gazebo/model_states', ModelStates)
        model_names = states.name
        object_pos = []
        object_names = []
        for i in range(len(model_names)):
            if model_names[i] not in self.scene_models:
                x = states.pose[i].position.x
                y = states.pose[i].position.y
                object_pos.append((x, y))
                object_names.append(model_names[i])
        return object_names, object_pos
    
    def get_center(self):
        object_names, object_pos = self.get_model_states()
        max_two_points = []
        max_dis = -1
        for i in range(len(object_pos)):
            for j in range(i + 1, len(object_pos)):
                distance = (object_pos[i][0] - object_pos[j][0]) ** 2 + (object_pos[i][1] - object_pos[j][1]) ** 2
                if distance > max_dis:
                    max_dis = distance
                    max_two_points = [object_pos[i], object_pos[j]]
        if not max_two_points:
            return -0.1, 0
        return (max_two_points[0][0] + max_two_points[1][0]) / 2, (max_two_points[0][1] + max_two_points[1][1]) / 2

    ######################################################################
    """Decision"""

    def grasp_strategy(self, data):
        if not self.robot_init:
            return
        index = 0
        grasp_position = data.grasp_position[index: index + 5]
        self.grasp_target.publish(grasp_position)
    
    def step(self, action):

        # l = 0.1
        # center_a, center_b = self.get_center()
        # direction = action // 4 * 12
        # if 0 <= direction <= 90:
        #     xd, yd = 1, 1
        # elif 90 < direction <= 180:
        #     xd, yd = 0, 1
        # elif 180 < direction <= 270:
        #     xd, yd = 0, 0
        # else:
        #     xd, yd = 1, 0
        # # end_len = action % 4 - 1
        # end_len = 3
        # dx, dy = np.fabs(np.cos(direction)) * l, np.fabs(np.sin(direction)) * l
        # if xd == 1:
        #     start_x = center_a - dx * 3
        #     end_x = center_a + dx * end_len
        # else:
        #     start_x = center_a + dx * 3
        #     end_x = center_a - dx * end_len
        # if yd == 1:
        #     start_y = center_b - dy * 3
        #     end_y = center_b + dy * end_len
        # else:
        #     start_y = center_b + dy * 3
        #     end_y = center_b - dy * end_len

        # start_x = self.limit_to_ws('x', start_x)
        # start_y = self.limit_to_ws('y', start_y)
        # end_x = self.limit_to_ws('x', end_x)
        # end_y = self.limit_to_ws('y', end_y)
        if action < 5:
            start_x = 0.3 + action * 0.1
            start_y = 0.3
            end_x = start_x
            end_y = -0.3
        else:
            start_x = 0.8
            start_y = -0.3 + 0.1 * (action - 5) 
            end_x = 0.2
            end_y = start_y

        self.push((start_x, start_y), (end_x, end_y))
        
        _, object = self.get_model_states()
        for p in object:
            x, y = p
            if x < 0.2 or x > 0.8 or y < -0.3 or y > 0.3:
                return True, 0  
        
        separate, now_separate = self.sim_grasp()
        reward = separate - self.reward
        self.reawrd = now_separate

        _, object = self.get_model_states()
        if not object:
            return True, reward
        else:
            return False, reward

    
        

    ######################################################################
    """Control"""

    def init_joint_state(self):
        move_group = self.arm
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
        move_group.go(joint_goal, wait=True)
        move_group.stop()
        self.grasp(0)
        self.robot_init = True

    def go_to_pose_goal(self, x, y, z, r):
        self.robot_init = False
        print(time(), "go to pose goal(x, y, z, r)", x, y, z, r)
        move_group = self.arm
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.position.x = x
        pose_goal.position.y = y
        pose_goal.position.z = z

        quat = self.matrix_to_quaternion(-r)

        pose_goal.orientation.x = quat[0]
        pose_goal.orientation.y = quat[1]
        pose_goal.orientation.z = quat[2]
        pose_goal.orientation.w = quat[3]

        current_pose = self.arm.get_current_pose().pose
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

    def move_line(self):
        x1, y1, z1, ox1, oy1, oz1, ow1 = map(float, input().split())
        x2, y2, z2, ox2, oy2, oz2, ow2 = map(float, input().split())

        start = Pose()
        end = Pose()

        start.position.x = x1
        start.position.y = y1
        start.position.z = z1
        start.orientation.x = ox1
        start.orientation.y = oy1
        start.orientation.z = oz1
        start.orientation.w = ow1

        end.position.x = x2
        end.position.y = y2
        end.position.z = z2
        end.orientation.x = ox2
        end.orientation.y = oy2
        end.orientation.z = oz2
        end.orientation.w = ow2

        fraction = 0.0
        maxtries = 100
        attempts = 0 
        MPos_succ = False

        self.print_poses([start, end])
        self.arm.set_start_state_to_current_state()

        while fraction < 1.0 and attempts < maxtries:
            (plan, fraction) = self.arm.compute_cartesian_path (
                                    [start, end],
                                    0.01,
                                    0.0,
                                    True)
            attempts += 1         

        if fraction == 1.0:
            self.arm.execute(plan)
            MPos_succ = True
            rospy.sleep(0.2)
        else:
            rospy.loginfo("Path planning failed")

        self.init_joint_state()
        return MPos_succ

    
    def print_poses(self, poses):
        for p in poses:
            print(p.position.x, p.position.y, p.position.z, 
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
    
    def simply_goal_pose(self, x, y, z, ox, oy, oz, ow):
        move_group = self.arm
        pose_goal = geometry_msgs.msg.Pose()

        pose_goal.position.x = x
        pose_goal.position.y = y
        pose_goal.position.z = z

        pose_goal.orientation.x = ox
        pose_goal.orientation.y = oy
        pose_goal.orientation.z = oz
        pose_goal.orientation.w = ox

        move_group.set_pose_target(pose_goal)
        plan = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()

    def go_to_start_point(self, p):
        self.arm.set_pose_target(p)
        plan = self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()

    def push(self, s, e):
        start = Pose()
        end = Pose()

        self.arm.set_goal_position_tolerance(0.001)
        self.arm.set_goal_orientation_tolerance(0.01)
        self.arm.set_max_acceleration_scaling_factor(0.8)
        self.arm.set_max_velocity_scaling_factor(1)

        start.position.x = s[0]
        start.position.y = s[1]
        start.position.z = 1.16
        start.orientation.x = 1

        end.position.x = e[0]
        end.position.y = e[1]
        end.position.z = 1.16
        end.orientation.x = 1

        self.go_to_start_point(start)

        fraction = 0.0
        maxtries = 100
        attempts = 0 
        MPos_succ = False

        # self.print_poses([start, end])
        self.arm.set_start_state_to_current_state()

        while fraction < 1.0 and attempts < maxtries:
            (plan, fraction) = self.arm.compute_cartesian_path (
                                    [start, end],
                                    0.01,
                                    0.0,
                                    True)
            attempts += 1         

        if fraction == 1.0:
            self.arm.execute(plan)
            MPos_succ = True
            rospy.sleep(0.2)
        else:
            rospy.loginfo("Path planning failed")
        self.init_joint_state()
        return MPos_succ
    
    def sim_grasp(self):
        object_names, object = self.get_model_states()
        not_separated_models = set()
        separate = 0
        for i in range(len(object)):
            if_separate = True
            for j in range(i + 1, len(object)):
                distance = (object[i][0] - object[j][0]) ** 2 + (object[i][1] - object[j][1]) ** 2
                if distance > 0.04:
                    separate += 1
                else:
                    if_separate = False
            if not if_separate:
                not_separated_models.add(i)
                not_separated_models.add(j)
        
        separated_models = set([i for i in range(len(object))]) - not_separated_models
        for i in separated_models:
            rospy.wait_for_service('/gazebo/delete_model', timeout=5)
            self.delete_model(object_names[i])
        return separate, separate - (len(object) - 1) * len(separated_models)

    ######################################################################
    """Robot Environment"""

    def add_box(self, timeout=4):
        box_name = self.box_name
        scene = self.scene

        rospy.sleep(1)
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = self.arm.get_planning_frame()
        box_pose.pose.orientation.w = 0
        box_pose.pose.orientation.x = 1.0
        box_pose.pose.orientation.y = 0
        box_pose.pose.orientation.z = 0
        box_pose.pose.position.z = 0.97
        box_name = "table"
        scene.add_box("table", box_pose, size=(2, 2, 0.01))

    def reset(self):
        # rospy.wait_for_service('/gazebo/reset_world', timeout=5)
        # self.reset_world()
        # rospy.sleep(3)
        self.init_joint_state()
        rospy.wait_for_service('/gazebo/delete_model', timeout=5)
        object_names, _ = self.get_model_states()
        for i in object_names:
            self.delete_model(i)
        self.object_spwan()
        _, self.reward = self.sim_grasp()
    
    def set_model_state(self, model_name):
        rospy.wait_for_service('/gazebo/set_model_state', timeout=5)
        state_msg = ModelState()
        state_msg.model_name = model_name
        state_msg.pose.position.x = 2
        state_msg.pose.position.y = 0
        state_msg.pose.position.z = 0.4
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 1
        self.set_state(state_msg)

    def object_spwan(self):
        # initial all objects, and spwan them to the Gazebo 
        rospy.wait_for_service("gazebo/spawn_sdf_model",timeout=5)

        for i in self.object_models:
            initial_pose = Pose()
            initial_pose.position.x = np.random.uniform(0.4,0.6)
            initial_pose.position.y = np.random.uniform(-0.1,0.1)
            initial_pose.position.z = 1.2
            
            initial_orientation = quaternion_from_euler(np.random.uniform(-3.14,3.14), np.random.uniform(-3.14,3.14), np.random.uniform(-3.14,3.14))
            initial_pose.orientation.x = initial_orientation[0]
            initial_pose.orientation.y = initial_orientation[1]
            initial_pose.orientation.z = initial_orientation[2]
            initial_pose.orientation.w = initial_orientation[3]

            filename = '/home/ljm/.gazebo/models/%s/model.sdf'%(i)
            with open(filename,"r") as f:
                reel_xml = f.read()
            self.spawn(i, reel_xml, "", initial_pose, "world")
        rospy.sleep(2)

    ######################################################################
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
    
    def limit_to_ws(self, axis, value):
        if axis == 'x':
            if value < 0.2:
                value = 0.2
            if value > 0.8:
                value = 0.8
        else:
            if value < -0.3:
                value = 0.3
            if value > 0.3:
                value = 0.3
        return value


def main():
    sim = RobotSim()
    func = int(sys.argv[1])
    if func == 0:
      sim.go_to_joint_state()
    elif func == 1:
      sim.simply_goal_pose(float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]),
       float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8]))
    elif func == 2:
      sim.grasp(float(sys.argv[2]))
    elif func == 3:
      sim.step(int(sys.argv[2]))
    elif func == 4:
      sim.move_line()
    elif func == 5:
      sim.reset()
    elif func == 6:
        for i in range(10):
            print(i)
            sim.step(i)

if __name__ == "__main__":
    main()
