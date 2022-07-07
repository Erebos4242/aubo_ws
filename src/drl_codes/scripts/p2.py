#!/usr/bin/env python3
import rospy
import os
from gazebo_msgs.srv import DeleteModel, SetModelState, SpawnModel
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelStates, ModelState


objects = ['cappuccino',
            'chewinggum',
            'eraser',
            'salt',
            'sugar',
            'sweetener',
            'teayellow',
            'z_carrs',
            'z_milk1',
            'z_orange']

delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
spawn = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)

for i in objects:
    # print(i)
    # object_names, _ = env.get_model_states()
    # for i in object_names:
    #     env.delete_model(i)

    initial_pose = Pose()
    initial_pose.position.x = 0
    initial_pose.position.y = 0
    initial_pose.position.z = 1

    initial_pose.orientation.x = 0
    initial_pose.orientation.y = 0
    initial_pose.orientation.z = 0
    initial_pose.orientation.w = 1

    filename = '/home/ljm/codes/myur_ws/src/ur_robotiq/ur_robotiq_gazebo/meshes/%s/model.sdf'%(i)
    with open(filename,"r") as f:
        reel_xml = f.read()
    spawn(i, reel_xml, "", initial_pose, "world")
    rospy.sleep(2)