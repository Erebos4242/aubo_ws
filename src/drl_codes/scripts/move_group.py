#!/usr/bin/env python3
from multiprocessing.connection import wait
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
import random
import time
from geometry_msgs.msg import PoseStamped, Pose
from copy import deepcopy
## END_SUB_TUTORIAL


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


class MoveGroupPythonIntefaceTutorial(object):
  """MoveGroupPythonIntefaceTutorial"""
  def __init__(self):
    super(MoveGroupPythonIntefaceTutorial, self).__init__()

    ## BEGIN_SUB_TUTORIAL setup
    ##
    ## First initialize `moveit_commander`_ and a `rospy`_ node:
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_group_python_interface_tutorial', anonymous=True)

    ## Instantiate a `RobotCommander`_ object. Provides information such as the robot's
    ## kinematic model and the robot's current joint states
    robot = moveit_commander.RobotCommander()

    ## Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
    ## for getting, setting, and updating the robot's internal understanding of the
    ## surrounding world:
    scene = moveit_commander.PlanningSceneInterface()

    ## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
    ## to a planning group (group of joints).  In this tutorial the group is the primary
    ## arm joints in the Panda robot, so we set the group's name to "panda_arm".
    ## If you are using a different robot, change this value to the name of your robot
    ## arm planning group.
    ## This interface can be used to plan and execute motions:
    group_name = "robot"
    move_group = moveit_commander.MoveGroupCommander(group_name)

    ## Create a `DisplayTrajectory`_ ROS publisher which is used to display
    ## trajectories in Rviz:
    display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                   moveit_msgs.msg.DisplayTrajectory,
                                                   queue_size=20)

    ## END_SUB_TUTORIAL

    ## BEGIN_SUB_TUTORIAL basic_info
    ##
    ## Getting Basic Information
    ## ^^^^^^^^^^^^^^^^^^^^^^^^^
    # We can get the name of the reference frame for this robot:
    planning_frame = move_group.get_planning_frame()


    # We can also print the name of the end-effector link for this group:
    eef_link = move_group.get_end_effector_link()


    # We can get a list of all the groups in the robot:
    group_names = robot.get_group_names()


    # Sometimes for debugging it is useful to print the entire state of the
    # robot:

    ## END_SUB_TUTORIAL

    # Misc variables
    self.box_name = ''
    self.robot = robot
    self.scene = scene
    self.move_group = move_group
    self.display_trajectory_publisher = display_trajectory_publisher
    self.planning_frame = planning_frame
    self.eef_link = eef_link
    self.group_names = group_names
    self.init_joint = [1.571, 0, 0, 0, 0, 0]


  def go_to_joint_state(self):
    # Copy class variables to local variables to make the web tutorials more clear.
    # In practice, you should use the class variables directly unless you have a good
    # reason not to.
    move_group = self.move_group

    ## BEGIN_SUB_TUTORIAL plan_to_joint_state
    ##
    ## Planning to a Joint Goal
    ## ^^^^^^^^^^^^^^^^^^^^^^^^
    ## The Panda's zero configuration is at a `singularity <https://www.quora.com/Robotics-What-is-meant-by-kinematic-singularity>`_ so the first
    ## thing we want to do is move it to a slightly better configuration.
    # We can get the joint values from the group and adjust some of the values:
    joint_goal = move_group.get_current_joint_values()
    joint_goal[0] = 0
    joint_goal[1] = -pi/4
    joint_goal[2] = 0
    joint_goal[3] = -pi/2
    joint_goal[4] = 0
    joint_goal[5] = pi/3
    joint_goal[6] = 0

    # The go command can be called with joint values, poses, or without any
    # parameters if you have already set the pose or joint target for the group
    move_group.go(joint_goal, wait=True)

    # Calling ``stop()`` ensures that there is no residual movement
    move_group.stop()

    ## END_SUB_TUTORIAL

    # For testing:
    current_joints = move_group.get_current_joint_values()
    return all_close(joint_goal, current_joints, 0.01)


  def go_to_pose_goal(self):
    # Copy class variables to local variables to make the web tutorials more clear.
    # In practice, you should use the class variables directly unless you have a good
    # reason not to.
    move_group = self.move_group

    ## BEGIN_SUB_TUTORIAL plan_to_pose
    ##
    ## Planning to a Pose Goal
    ## ^^^^^^^^^^^^^^^^^^^^^^^
    ## We can plan a motion for this group to a desired pose for the
    ## end-effector:
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.orientation.w = 0
    pose_goal.position.x = -0.3
    pose_goal.position.y = 0.3
    pose_goal.position.z = 1.3

    move_group.set_pose_target(pose_goal)

    ## Now, we call the planner to compute the plan and execute it.
    plan = move_group.go(wait=True)
    # Calling `stop()` ensures that there is no residual movement
    move_group.stop()
    # It is always good to clear your targets after planning with poses.
    # Note: there is no equivalent function for clear_joint_value_targets()
    move_group.clear_pose_targets()

    # joint_goal = move_group.get_current_joint_values()
    # joint_goal[4] = joint_goal[4] - pi/2
    # move_group.go(joint_goal, wait=True)
    # move_group.stop()
    
    ## END_SUB_TUTORIAL

    # For testing:
    # Note that since this section of code will not be included in the tutorials
    # we use the class variable rather than the copied state variable
    current_pose = self.move_group.get_current_pose().pose
    return all_close(pose_goal, current_pose, 0.01)

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
    move_group.go(joint_goal, wait=True)
    move_group.stop()
    self.robot_init = True

  def Moveit_pose(self,*pose):   
        # 设置机器人终端的目标位置
        # 姿态使用四元数描述，基于base_link坐标系                                   
        target_pose = PoseStamped()
        target_pose.header.frame_id = 'world'
        target_pose.header.stamp = rospy.Time.now()  #记录此时的时间（时间戳） 
        target_pose.pose.position.x = 0.121530
        target_pose.pose.position.y = -0.465707
        target_pose.pose.position.z = 0.509740
        target_pose.pose.orientation.x = 1
        target_pose.pose.orientation.y = 0
        target_pose.pose.orientation.z = 0
        target_pose.pose.orientation.w = 0
        # 设置机械臂终端运动的目标位姿
        self.move_group.set_pose_target(target_pose, 'wrist3_Link')
        # 初始化路点列表
        waypoints = []
        # 将初始位姿加入路点列表
        waypoints.append(target_pose.pose)
        wpose = deepcopy(target_pose.pose)
        # 计算每次移动后的运动坐标，放置在运动列表中保存下来。
        wpose.position.x = self.pose1[0]
        wpose.position.y = self.pose1[1]
        wpose.position.z = self.pose1[2]
        waypoints.append(deepcopy(wpose))
        wpose.position.x = self.pose2[0]
        wpose.position.y = self.pose2[1]
        wpose.position.z = self.pose2[2]
        waypoints.append(deepcopy(wpose))
        
        # 笛卡尔空间下的路径规划  
        fraction = 0.0   #路径规划覆盖率
        maxtries = 100   #最大尝试规划次数
        attempts = 0     #已经尝试规划次数
        # 设置机器臂当前的状态作为运动初始状态
        self.move_group.set_start_state_to_current_state()
        # 尝试规划一条笛卡尔空间下的路径，依次通过所有路点，完成圆弧轨迹
        while fraction < 1.0 and attempts < maxtries:
            (plan, fraction) = self.move_group.compute_cartesian_path (
                                    waypoints,   # waypoint poses，路点列表
                                    0.01,        # eef_step，终端步进值
                                    0.0,         # jump_threshold，跳跃阈值
                                    True)        # avoid_collisions，避障规划
            # 尝试次数累加
            attempts += 1
            # 打印运动规划进程
            if attempts % 10 == 0:
                rospy.loginfo("Still trying after " + str(attempts) + " attempts...")            
        # 如果路径规划成功（覆盖率100%）,则开始控制机械臂运动
        if fraction == 1.0:
            rospy.loginfo("Path computed successfully. Moving the arm.")
            self.move_group.execute(plan)
        # 如果路径规划失败，则打印失败信息
        else:
            rospy.loginfo("Path planning failed with only " + str(fraction) + " success after " + str(maxtries) + " attempts.")  
        rospy.sleep(1)

  def chose(self, action):
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
    self.pose1 = start_x, start_y, 1.16
    self.pose2 = end_x, end_y, 1.16

  def move_line(self, s, e):
    start = geometry_msgs.msg.Pose()
    end = geometry_msgs.msg.Pose()

    start.position.x = s[0]
    start.position.y = s[1]
    start.position.z = 1.16
    start.orientation.x = 1
    start.orientation.y = 0
    start.orientation.z = 0
    start.orientation.w = 0

    end.position.x = e[0]
    end.position.y = e[1]
    end.position.z = 1.16
    end.orientation.x = 1
    end.orientation.y = 0
    end.orientation.z = 0
    end.orientation.w = 0


    # self.print_poses([start, end])

    (plan, fraction) = self.move_group.compute_cartesian_path(
        [start, end],  # waypoints to follow
        0.01,  # eef_step
        0.0)  # jump_threshold
    
    self.move_group.execute(plan, wait=True)

    self.init_joint_state()


if __name__ == '__main__':
  env = MoveGroupPythonIntefaceTutorial()
  for i in range(100):
    stime = time.time()
    action = random.randint(0, 9)
    env.chose(action)
    env.Moveit_pose()
    etime = time.time()
    print(f'count: {i}, actions: {action}, time cost: {etime - stime}')
    rospy.sleep(0.1)