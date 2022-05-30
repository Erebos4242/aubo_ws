#!/usr/bin/env python3
from curses.ascii import SI
from multiprocessing.connection import wait
import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from moveit_commander.conversions import pose_to_list
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
from time import time
import threading


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

        """publisher"""
        display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                       moveit_msgs.msg.DisplayTrajectory,
                                                       queue_size=20)
        grasp_position_map_pub = rospy.Publisher('/grasp/map', Image, queue_size=10)
        grasp_position_pub = rospy.Publisher('/grasp/position', GraspPosition, queue_size=1)
        graper_command = rospy.Publisher('/aubo_i5/graper_position_controller/command', Float64, queue_size=10)
        grasp_target = rospy.Publisher('/grasp/target', GraspPosition, queue_size=1)

        """publisher"""
        self.display_trajectory_publisher = display_trajectory_publisher
        self.graper_command = graper_command
        self.grasp_position_map_pub = grasp_position_map_pub
        self.grasp_position_pub = grasp_position_pub
        self.grasp_target = grasp_target

        """init env"""
        # self.add_box()

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
        move_group.go(joint_goal, wait=True)
        move_group.stop()
        self.robot_init = True
    
    def print_poses(self, poses):
        for p in poses:
            print(p.position.x, p.position.y, p.position.z, 
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)

    def push(self, s, e):
        start = geometry_msgs.msg.Pose()
        end = geometry_msgs.msg.Pose()

        start.position.x = s[0]
        start.position.y = s[1]
        start.position.z = 1.16
        start.orientation.y = 1

        end.position.x = e[0]
        end.position.y = e[1]
        end.position.z = 1.16
        end.orientation.y = 1

        self.print_poses([start, end])

        (plan, fraction) = self.move_group.compute_cartesian_path(
            [start, end],  # waypoints to follow
            0.01,  # eef_step
            0.0)  # jump_threshold
        
        self.move_group.execute(plan, wait=True)

        self.init_joint_state()
    
    def step(self, action):
        if action < 5:
            start_x = 0.3 + action * 0.1
            start_y = 0.3
            end_x = start_x
            end_y = -0.3
        else:
            start_x = 0.8
            start_y = 0.4 - 0.1 * action 
            end_x = 0.2
            end_y = start_y
        self.push([start_x, start_y], [end_x, end_y])
        

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


if __name__ == "__main__":
    env = RobotSim()
    env.step(int(sys.argv[1]))
