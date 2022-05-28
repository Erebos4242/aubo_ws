#!/usr/bin/env python3
from itertools import starmap
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from std_msgs.msg import Float64
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from time import time
import numpy as np  
# from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import DeleteModel, SetModelState, SpawnModel
from gazebo_msgs.msg import ModelStates, ModelState
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
  """MoveGroupPythonIntefaceTutorial"""
  def __init__(self):
    super(RobotSim, self).__init__()

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('RobotSim', anonymous=True)
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()

    group_name = "robot"
    # group_name = "manipulator_i5"
    move_group = moveit_commander.MoveGroupCommander(group_name)
    display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                   moveit_msgs.msg.DisplayTrajectory,
                                                   queue_size=20)

    planning_frame = move_group.get_planning_frame()
    eef_link = move_group.get_end_effector_link()

    group_names = robot.get_group_names()

    graper_command = rospy.Publisher('/aubo_i5/graper_position_controller/command', Float64, queue_size=10)

    self.box_name = 'table_surface'
    self.robot = robot
    self.scene = scene
    self.move_group = move_group
    self.display_trajectory_publisher = display_trajectory_publisher
    self.planning_frame = planning_frame
    self.eef_link = eef_link
    self.group_names = group_names
    self.init_joint = [1.571, 0, 0, 0, 0, 0]
    self.add_box()
    self.ready = True
    self.init_pose = move_group.get_current_pose().pose

    self.scene_models = ['ground_plane', 'table_marble', 'kinect', 'background', 'aubo_i5']
    self.object_models = ['cleaner', 'eraser', 'shampoo', 'chewinggum', 'salt']

    self.graper_command = graper_command
    self.delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
    self.spawn = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)

  def go_to_joint_state(self):
    move_group = self.move_group
    joint_goal = move_group.get_current_joint_values()
    joint_goal[0] = self.init_joint[0]
    joint_goal[1] = self.init_joint[1]
    joint_goal[2] = self.init_joint[2]
    joint_goal[3] = self.init_joint[3]
    joint_goal[4] = self.init_joint[4]
    joint_goal[5] = self.init_joint[5]
    move_group.go(joint_goal, wait=True)
    move_group.stop()
    current_joints = move_group.get_current_joint_values()
    return all_close(joint_goal, current_joints, 0.01)


  def go_to_pose_goal(self, x, y, z, r):
    self.ready = False
    print(time(), "go_to_pose_goal(x, y, z, r)", x, y, z, r)
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

    move_group.set_pose_target(pose_goal)
    plan = move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()
    current_pose = self.move_group.get_current_pose().pose
    print(time(), "go_to_pose_goal", "complete")
    # rospy.sleep(5)
    # print(time(), "init_joint")
    # self.go_to_joint_state()
    # print(time(), "init_joint", "complete")
    self.ready = True
    return all_close(pose_goal, current_pose, 0.01)

  def simply_goal_pose(self, x, y, z, ox, oy, oz, ow):
    move_group = self.move_group
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

  def grasp(self, data):
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

  def mult_between_quaternions(self, right_qua, left_qua):
      x = right_qua[3] * left_qua[0] + left_qua[3] * right_qua[0] + (
                  right_qua[1] * left_qua[2] - right_qua[2] * left_qua[1])
      y = right_qua[3] * left_qua[1] + left_qua[3] * right_qua[1] - (
                  right_qua[0] * left_qua[2] - right_qua[2] * left_qua[0])
      z = right_qua[3] * left_qua[2] + left_qua[3] * right_qua[2] + (
                  right_qua[0] * left_qua[1] - right_qua[1] * left_qua[0])
      w = right_qua[3] * left_qua[3] - (
                  right_qua[0] * left_qua[0] + right_qua[1] * left_qua[1] + right_qua[2] * left_qua[2])
      return np.round(x, 15), np.round(y, 15), np.round(z, 15), np.round(w, 15)

  def rotation_to_quaternion(self, axis, angle):  # return x, y, z, w
      angle /= 2
      cos_num = np.cos(angle)
      sin_num = np.sin(angle)
      if axis == 'x':
          return sin_num, 0, 0, cos_num
      elif axis == 'y':
          return 0, sin_num, 0, cos_num
      else:
          return 0, 0, sin_num, cos_num

  def rotation_integrate(self, quaternion_list):
      if not quaternion_list:
          return
      if len(quaternion_list) == 1:
          return quaternion_list[0]
      res = self.mult_between_quaternions(quaternion_list[1], quaternion_list[0])
      for q in quaternion_list[2:]:
          res = self.mult_between_quaternions(q, res)
      return res

  def print_poses(self, poses):
      for p in poses:
        print(p.position.x, p.position.y, p.position.z, 
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)

  def path_plan(self, scale=1):
      move_group = self.move_group
      waypoints = []
      pose = move_group.get_current_pose().pose
      pose.position.x = -0.3
      pose.position.y = 0.3
      pose.position.z = 1.5

      qua = self.rotation_to_quaternion('x', np.pi / 2)

      pose.orientation.x = qua[0]
      pose.orientation.y = qua[1]
      pose.orientation.z = qua[2]
      pose.orientation.w = qua[3]

      y = 0.3
      while y >= -0.3:
          x = -(0.09 - y ** 2) ** 0.5
          qua_after = self.rotation_to_quaternion('y', np.arctan(abs(x / y)))
          qua = self.rotation_integrate([qua_after, qua])
          pose.position.x = x
          pose.position.y = y
          pose.orientation.x = qua[0]
          pose.orientation.y = qua[1]
          pose.orientation.z = qua[2]
          pose.orientation.w = qua[3]
          waypoints.append(copy.deepcopy(pose))
          y -= 0.05

      self.print_poses(waypoints)

      (plan, fraction) = move_group.compute_cartesian_path(
          waypoints,  # waypoints to follow
          0.01,  # eef_step
          0.0)  # jump_threshold
      return plan, fraction
  
  def move_line(self):
    start = geometry_msgs.msg.Pose()
    end = geometry_msgs.msg.Pose()

    start.position.x = 0.3
    start.position.y = 0.3
    start.position.z = 1.16
    start.orientation.y = 1

    end.position.x = 0.7
    end.position.y = -0.3
    end.position.z = 1.16
    end.orientation.y = 1

    self.print_poses([start, end])

    (plan, fraction) = self.move_group.compute_cartesian_path(
          [start, end],  # waypoints to follow
          0.01,  # eef_step
          0.0)  # jump_threshold


    # points = []

    # start.position.x = 0.2
    # start.position.y = 0.3
    # start.position.z = 1.16
    # start.orientation.x = 1

    # points.append(start)

    # start.position.x += 0.6
    # points.append(copy.deepcopy(start))

    # start.position.y -= 0.6
    # points.append(copy.deepcopy(start))

    # start.position.x -= 0.6
    # points.append(copy.deepcopy(start))

    # start.position.y += 0.6
    # points.append(copy.deepcopy(start))

    # self.print_poses(points)

    # (plan, fraction) = self.move_group.compute_cartesian_path(
    #   points,  # waypoints to follow
    #   0.01,  # eef_step
    #   0.0)  # jump_threshold


    self.execute_plan(plan)

  def execute_plan(self, plan):
    move_group = self.move_group
    move_group.execute(plan, wait=True)

  def execute_path(self):
    move_group = self.move_group
    waypoints = []
    pose = move_group.get_current_pose().pose
    pose.position.x = -0.3
    pose.position.y = 0.3
    pose.position.z = 1.5

    qua = self.rotation_to_quaternion('x', np.pi / 2)

    pose.orientation.x = qua[0]
    pose.orientation.y = qua[1]
    pose.orientation.z = qua[2]
    pose.orientation.w = qua[3]

    y = 0.3
    while y >= -0.3:
        x = -(0.09 - y ** 2) ** 0.5
        qua_after = self.rotation_to_quaternion('y', np.arctan(abs(x / y)))
        qua = self.rotation_integrate([qua_after, qua])
        pose.position.x = x
        pose.position.y = y
        pose.orientation.x = qua[0]
        pose.orientation.y = qua[1]
        pose.orientation.z = qua[2]
        pose.orientation.w = qua[3]
        waypoints.append(copy.deepcopy(pose))
        y -= 0.05

    self.print_poses(waypoints)

    for p in waypoints:
      move_group.set_pose_target(p)
      plan = move_group.go(wait=True)
      move_group.stop()
      move_group.clear_pose_targets()

  def reset(self):
    # rospy.wait_for_service('/gazebo/reset_world', timeout=5)
    # self.reset_world()
    # rospy.sleep(3)
    rospy.wait_for_service('/gazebo/delete_model', timeout=5)
    object_names, _ = self.get_model_states()
    for i in object_names:
        self.delete_model(i)
    self.object_spwan()
  
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

  def step(self, action):
    while True:
      self.move_line()


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
      plan, fraction = sim.path_plan()
      sim.execute_plan(plan)
    elif func == 4:
      while True:
        sim.move_line()
    elif func == 5:
      sim.reset()


if __name__ == "__main__":
    main()