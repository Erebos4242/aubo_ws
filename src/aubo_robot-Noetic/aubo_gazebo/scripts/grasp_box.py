#!/usr/bin/env python3
import rospy
from aubo_msgs.msg import GraspPosition
import numpy as np
import cv2
from aubo_gazebo_sim import *
from time import time


robot_ready = True


def grasp_excution(data):
    global robot_ready

    robot_ready = False
    target = data.grasp_position

    if aubo_sim.ready:
        aubo_sim.go_to_pose_goal(target[0], target[1], target[2] + 0.2, target[3])

    robot_ready = True



def grasp_strategy(data):
    global robot_ready
    if not robot_ready:
        return
    index = 0
    grasp_position = data.grasp_position[index: index + 4]
    grasp_target.publish(grasp_position)


def main():
    # rospy.init_node('grasp_box')
    rospy.Subscriber("/grasp/position", GraspPosition, grasp_strategy)
    rospy.Subscriber('/grasp/target', GraspPosition, grasp_excution)
    rospy.spin()
    

if __name__ == "__main__":
    aubo_sim = RobotSim()
    grasp_target = rospy.Publisher('/grasp/target', GraspPosition, queue_size=1)
    main()