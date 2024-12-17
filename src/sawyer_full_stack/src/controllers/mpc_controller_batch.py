#!/usr/bin/env python

import rospy
import rospkg
import roslaunch
import numpy as np
import cvxpy as cp
from intera_interface import Limb
import matplotlib.pyplot as plt

from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from intera_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest

import tf2_ros
from moveit_msgs.msg import DisplayTrajectory, RobotState
from sawyer_pykdl import sawyer_kinematics

from geometry_msgs.msg import Point, PointStamped

def batch_solver(initial_position, target_position, steps, delta_t):
    """
    Calculate a feasible trajectory using a batch approach.
    """
    n_joints = len(initial_position)
    X = cp.Variable((n_joints, steps + 1))  # Joint positions
    U = cp.Variable((n_joints, steps + 1))  # Joint velocities

    cost = 0
    constraints = [X[:, 0] == initial_position]
    global maxA
    global maxV
    maxA = np.array([3.5, 2.5, 5.0, 5.0, 5.0, 5.0, 5.0])
    maxV = np.array([0.88, 0.678, 0.996, 0.996, 1.776, 1.776, 2.316])
    # maxA = np.array([7.0, 5.0, 8.0, 8.0, 8.0, 8.0, 8.0])
    # maxV = np.array([1.48, 1.13, 1.66, 1.66, 2.96, 2.96, 2.86])

    for k in range(steps):
        cost += cp.norm(X[:, k] - target_position, 2)  # Minimize tracking error
        constraints += [X[:, k + 1] == X[:, k] + delta_t * U[:, k]]  # Dynamics
        constraints += [U[:, k] >= -1 * maxV, U[:, k] <= maxV]
        constraints += [U[:, k + 1] <= U[:, k] + delta_t * maxA, U[:, k + 1] >= U[:, k] - delta_t * maxA]

    # Terminal cost
    cost += cp.norm(X[:, steps] - target_position, 2)

    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    return X.value, U.value  # Return positions and velocities

def mpc_optimization(current_position, reference_trajectory, horizon=20, delta_t=0.1):
    """
    Perform MPC to compute joint velocity commands for tracking the reference trajectory.
    """
    n_joints = len(current_position)
    U = cp.Variable((n_joints, horizon + 1))
    X = cp.Variable((n_joints, horizon + 1))

    cost = 0
    constraints = [X[:, 0] == current_position]
    # maxA = np.array([3.5, 2.5, 5.0, 5.0, 5.0, 5.0, 5.0])
    # maxV = np.array([0.88, 0.678, 0.996, 0.996, 1.776, 1.776, 2.316])
    global maxA
    global maxV

    for k in range(horizon):
        cost += cp.norm(X[:, k] - reference_trajectory[:, k], 2)  # Track reference trajectory
        cost += 0.2 * cp.norm(U[:, k], 2)  # Penalize control effort
        constraints += [X[:, k + 1] == X[:, k] + delta_t * U[:, k]]
        constraints += [U[:, k] >= -1 * maxV, U[:, k] <= maxV]
        constraints += [U[:, k + 1] <= U[:, k] + delta_t * maxA, U[:, k + 1] >= U[:, k] - delta_t * maxA]

    # Terminal cost
    cost += cp.norm(X[:, horizon] - reference_trajectory[:, horizon], 2)

    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    return U.value[:, 0]  # Return the first velocity command

def plot_joint_data(time_data, joint_states, joint_velocities, joint_names):
    """
    Plot joint states and velocities over time.
    """
    plt.figure(figsize=(12, 10))

    # Plot joint states
    plt.subplot(2, 1, 1)
    for joint in joint_names:
        plt.plot(time_data, joint_states[joint], label=joint)
    plt.title("Joint States Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint Position (radians)")
    plt.legend()
    plt.grid()

    # Plot joint velocities
    plt.subplot(2, 1, 2)
    for joint in joint_names:
        plt.plot(time_data, joint_velocities[joint], label=joint)
    plt.title("Joint Velocities Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint Velocity (radians/s)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

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

def goal_point_callback(msg):
    """
    Callback function for receiving 3D goal point from the "goal_point" topic.
    """
    global goal_position
    goal_position = np.array([msg.x, msg.y, msg.z])
    print("Received Goal Point:", goal_position)

def lookup_tag(tag_number=0):
        tfBuffer = tf2_ros.Buffer()
        tfListener = tf2_ros.TransformListener(tfBuffer)
        try:
            trans = tfBuffer.lookup_transform('base', f'ar_marker_{tag_number}', rospy.Time(0), rospy.Duration(10.0))
        except Exception as e:
            print(e)
            print("Retrying ...")
        tag_pos = [getattr(trans.transform.translation, dim) for dim in ('x', 'y', 'z')]
        return tag_pos

def ik_service_client(goal_pos, *args, **kwargs):
    """IK solver from Lab 5"""
    service_name = "ExternalTools/right/PositionKinematicsNode/IKService"
    ik_service_proxy = rospy.ServiceProxy(service_name, SolvePositionIK)
    ik_request = SolvePositionIKRequest()
    header = Header(stamp=rospy.Time.now(), frame_id='base')

    pose_stamped = PoseStamped()
    pose_stamped.header = header

    pose_stamped.pose.position.x = goal_pos[0]
    pose_stamped.pose.position.y = goal_pos[1]
    pose_stamped.pose.position.z = goal_pos[2]

    pose_stamped.pose.orientation.x = 0
    pose_stamped.pose.orientation.y = 0.71
    pose_stamped.pose.orientation.z = 0
    pose_stamped.pose.orientation.w = 0.71

    ik_request.pose_stamp.append(pose_stamped)
    ik_request.tip_names.append('right_hand')

    try:
        rospy.wait_for_service(service_name, 5.0)
        response = ik_service_proxy(ik_request)
    except (rospy.ServiceException, rospy.ROSException) as e:
        rospy.logerr("Service call failed: %s" % (e,))
        return

    if (response.result_type[0] > 0):
        return np.array(response.joints[0].position)
    return None

def main():
    """
    Main control loop to run the MPC controller and collect data for plotting.
    """
    rospy.init_node('mpc_sawyer_controller', anonymous=True)
    limb = Limb('right')
    rate = rospy.Rate(10)
    delta_t = 0.1

    # Subscribe to the goal_point topic to get the 3D goal position
    rospy.Subscriber("/strike_point", Point, goal_point_callback)
    
    while 'goal_position' not in globals():
        pass
        # print("waiting for strike point...")

    # Convert goal position to goal joint angles with inverse kinematics
    global goal_position
    print("Goal Position: ", goal_position)
    target_position = ik_service_client(goal_position)
    print(target_position)

    current_position = np.array([-0.0003447265625, -0.9987998046875, -0.0013974609375,
                                  1.4995615234375, -0.0002099609375, -0.5000400390625,
                                  1.6983779296875])
    # target_position = target_pos
    # target_position = np.array([-0.5, -1.5, -0.2, 1.5, -0.05, -1, 1])

    steps = 50
    batch_positions, batch_velocities = batch_solver(current_position, target_position, steps, delta_t)

    rospy.loginfo("Starting MPC control loop...")

    time_data = []
    joint_states = {joint: [] for joint in limb.joint_names()}
    joint_velocities = {joint: [] for joint in limb.joint_names()}
    start_time = rospy.get_time()

    mpc_horizon = 20
    for k in range(steps - mpc_horizon):
        reference_trajectory = batch_positions[:, k:k + mpc_horizon + 1]
        current_state = np.array([limb.joint_angle(j) for j in limb.joint_names()])

        control_input = mpc_optimization(current_state, reference_trajectory)

        joint_command = dict(zip(limb.joint_names(), control_input))
        limb.set_joint_velocities(joint_command)

        current_time = rospy.get_time() - start_time
        time_data.append(current_time)
        for idx, joint in enumerate(limb.joint_names()):
            joint_states[joint].append(current_state[idx])
            joint_velocities[joint].append(control_input[idx])

        rospy.loginfo(f"Current State: {current_state}")
        rospy.loginfo(f"Control Input: {control_input}")

        if np.linalg.norm(current_state - target_position) < 0.4:
            joint_command = dict(zip(limb.joint_names(), np.array([0, 0, 0, 0, 0, 0, 0])))
            limb.set_joint_velocities(joint_command)
            rospy.loginfo("Target reached!")
            rospy.loginfo(current_state - target_position)
            rospy.loginfo(np.linalg.norm(current_state - target_position))
            break

        rate.sleep()

    plot_joint_data(time_data, joint_states, joint_velocities, limb.joint_names())

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("MPC control loop interrupted.")
