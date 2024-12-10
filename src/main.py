#!/usr/bin/env python
"""
Starter script for 106a lab7. 
Author: Chris Correa
"""
import sys
import argparse
import numpy as np
import rospkg
import roslaunch

from planning.trajectories import LinearTrajectory, CircularTrajectory
from planning.paths import MotionPath
from planning.path_planner import PathPlanner
from control.controllers import ( 
    PIDJointVelocityController, 
    FeedforwardJointVelocityController
)
from utils import *

from trac_ik_python.trac_ik import IK

import rospy
import tf2_ros
import intera_interface
from moveit_msgs.msg import DisplayTrajectory, RobotState
from sawyer_pykdl import sawyer_kinematics
from geometry_msgs.msg import Point


def tuck():
    """
    Tuck the robot arm to the start position. Use with caution
    """
    if input('Would you like to tuck the arm? (y/n): ') == 'y':
        rospack = rospkg.RosPack()
        path = rospack.get_path('sawyer_full_stack')
        launch_path = path + '/launch/custom_sawyer_tuck.launch'
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        launch = roslaunch.parent.ROSLaunchParent(uuid, [launch_path])
        launch.start()
    else:
        print('Canceled. Not tucking the arm.')


def get_trajectory(limb, kin, ik_solver, goal_pos, args):
    """
    Returns an appropriate robot trajectory for the specified task.  
    Now using goal_pos obtained from the "goal_point" topic.
    
    Parameters
    ----------
    task : string
        name of the task.  Options: line, circle, square
    goal_pos : 3x' :obj:`numpy.ndarray`
        The position obtained from the "goal_point" topic
    
    Returns
    -------
    :obj:`moveit_msgs.msg.RobotTrajectory`
    """
    num_way = args.num_way
    task = args.task

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    try:
        trans = tfBuffer.lookup_transform('base', 'right_hand', rospy.Time(0), rospy.Duration(10.0))
    except Exception as e:
        print(e)

    current_position = np.array([getattr(trans.transform.translation, dim) for dim in ('x', 'y', 'z')])
    print("Current Position:", current_position)

    if task == 'line':
        target_pos = goal_pos
        target_pos[2] += 0.4  # linear path moves to a Z position above goal
        print("TARGET POSITION:", target_pos)
        trajectory = LinearTrajectory(start_position=current_position, goal_position=target_pos, total_time=9)
    elif task == 'circle':
        target_pos = goal_pos
        target_pos[2] += 0.5
        print("TARGET POSITION:", target_pos)
        trajectory = CircularTrajectory(center_position=target_pos, radius=0.1, total_time=15)

    else:
        raise ValueError('task {} not recognized'.format(task))
    
    path = MotionPath(limb, kin, ik_solver, trajectory)
    return path.to_robot_trajectory(num_way, True)


def get_controller(controller_name, limb, kin):
    """
    Gets the correct controller from controllers.py

    Parameters
    ----------
    controller_name : string

    Returns
    -------
    :obj:`Controller`
    """
    if controller_name == 'open_loop':
        controller = FeedforwardJointVelocityController(limb, kin)
    elif controller_name == 'pid':
        Kp = 0.2 * np.array([0.4, 2, 1.7, 1.5, 2, 2, 3])
        Kd = 0.01 * np.array([2, 1, 2, 0.5, 0.8, 0.8, 0.8])
        Ki = 0.01 * np.array([1.4, 1.4, 1.4, 1, 0.6, 0.6, 0.6])
        Kw = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        controller = PIDJointVelocityController(limb, kin, Kp, Ki, Kd, Kw)
    else:
        raise ValueError('Controller {} not recognized'.format(controller_name))
    return controller


# New callback function for subscribing to goal_point
def goal_point_callback(msg):
    """
    Callback function for receiving 3D goal point from the "goal_point" topic.
    """
    global goal_position
    goal_position = np.array([msg.x, msg.y, msg.z])
    print("Received Goal Point:", goal_position)


def main():
    """
    Main function to plan and execute the trajectory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', '-t', type=str, default='line', help=
        'Options: line, circle. Default: line'
    )
    parser.add_argument('-controller_name', '-c', type=str, default='moveit', 
        help='Options: moveit, open_loop, pid. Default: moveit'
    )
    parser.add_argument('-rate', type=int, default=200, help="Control loop rate. Default: 200")
    parser.add_argument('-timeout', type=int, default=None, help="Timeout for the controller. Default: None")
    parser.add_argument('-num_way', type=int, default=50, help="Number of waypoints for trajectory. Default: 50")
    parser.add_argument('--log', action='store_true', help="Log controller performance")
    args = parser.parse_args()

    rospy.init_node('moveit_node')
    
    tuck()
    
    # this is used for sending commands (velocity, torque, etc) to the robot
    ik_solver = IK("base", "right_gripper_tip")
    limb = intera_interface.Limb("right")
    kin = sawyer_kinematics("right")

    # Subscribe to the goal_point topic to get the 3D goal position
    rospy.Subscriber("/goal_point", Point, goal_point_callback)
    
    # Wait for the first goal point to be received
    rospy.sleep(1)  # Allow some time for goal_point to be received
    
    if 'goal_position' not in globals():
        rospy.logerr("No goal position received from 'goal_point' topic.")
        sys.exit(1)

    # Get the robot trajectory based on the received goal position
    robot_trajectory = get_trajectory(limb, kin, ik_solver, goal_position, args)

    # This is a wrapper around MoveIt! for you to use.  We use MoveIt! to go to the start position
    planner = PathPlanner('right_arm')
    
    # By publishing the trajectory to the move_group/display_planned_path topic, you should 
    # be able to view it in RViz.  You will have to click the "loop animation" setting in 
    # the planned path section of MoveIt! in the menu on the left side of the screen.
    pub = rospy.Publisher('move_group/display_planned_path', DisplayTrajectory, queue_size=10)
    disp_traj = DisplayTrajectory()
    disp_traj.trajectory.append(robot_trajectory)
    disp_traj.trajectory_start = RobotState()
    pub.publish(disp_traj)

    # Move to the trajectory start position
    plan = planner.plan_to_joint_pos(robot_trajectory.joint_trajectory.points[0].positions)
    if args.controller_name != "moveit":
        plan = planner.retime_trajectory(plan, 0.3)
    planner.execute_plan(plan[1])

    if args.controller_name == "moveit":
        try:
            input('Press <Enter> to execute the trajectory using MOVEIT')
        except KeyboardInterrupt:
            sys.exit()
        # Uses MoveIt! to execute the trajectory.
        planner.execute_plan(robot_trajectory)
    else:
        controller = get_controller(args.controller_name, limb, kin)
        try:
            input('Press <Enter> to execute the trajectory using YOUR OWN controller')
        except KeyboardInterrupt:
            sys.exit()
        # execute the path using your own controller.
        done = controller.execute_path(
            robot_trajectory, 
            rate=args.rate, 
            timeout=args.timeout, 
            log=args.log
        )
        if not done:
            print('Failed to move to position')
            sys.exit(0)


if __name__ == "__main__":
    main()
