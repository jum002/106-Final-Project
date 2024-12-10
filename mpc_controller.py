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


def mpc_optimization(current_position, target_position, horizon=20, delta_t=0.1):
    """
    Perform MPC to compute joint velocity commands for smooth movement with better handling for joint 6.
    """
    n_joints = len(current_position)

    # Decision variables
    U = cp.Variable((n_joints, horizon + 1))  # Joint velocities
    X = cp.Variable((n_joints, horizon + 1))  # Predicted joint positions

    # Cost function
    cost = 0
    constraints = [X[:, 0] == current_position]
    maxA = np.array([3.5, 2.5, 5.0, 5.0, 5.0, 5.0, 5.0])
    maxV = np.array([0.88, 0.678, 0.996, 0.996, 1.776, 1.776, 2.316])

    for k in range(horizon):
        cost += cp.norm(X[:, k] - target_position, 2)  # Minimize tracking error
        cost += 0.2 * cp.norm(U[:, k], 2)  # Penalize control effort

        # # Smoother velocity for joint 6
        # if k < horizon - 1:
        #     cost += 0.05 * cp.norm(U[:, k + 1] - U[:, k], 2)  # Penalize velocity changes

        constraints += [X[:, k + 1] == X[:, k] + delta_t * U[:, k]]# Dynamics
        # constraints += [U[:, k] >= -1.0, U[:, k] <= 1.0]  # General velocity limits
        constraints += [U[:,k] >= -1*maxV, U[:,k] <= maxV]
        constraints +=[U[:, k + 1] <= U[:,k] + delta_t * maxA, U[:, k + 1] >= U[:,k] - delta_t * maxA]
        # constraints += [U[6, k] >= -0.4, U[6, k] <= 0.4]  # Tighter limits for joint 6

    # Add terminal cost for stabilization
    cost += cp.norm(X[:, horizon] - target_position, 2)  # Terminal stabilization

    # Define the optimization problem
    problem = cp.Problem(cp.Minimize(cost), constraints)

    # Solve the problem
    problem.solve()
    print("MPC_MPC_lol")
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



# New callback function for subscribing to goal_point
def goal_point_callback(msg):
    """
    Callback function for receiving 3D goal point from the "goal_point" topic.
    """
    global goal_position
    goal_position = np.array([msg.x, msg.y, msg.z])
    print("Received Goal Point:", goal_position)


def ik_service_client(goal_pos, *args, **kwargs):
        """IK solver from Lab 5

        Args:
            final_pose (List[7]): xyz, quat for final position

        Returns:
            np.ndarray: joint angles for the robot
        """
        service_name = "ExternalTools/right/PositionKinematicsNode/IKService"
        ik_service_proxy = rospy.ServiceProxy(service_name, SolvePositionIK)
        ik_request = SolvePositionIKRequest()
        header = Header(stamp=rospy.Time.now(), frame_id='base')

        # Create a PoseStamped and specify header (specifying a header is very important!)
        pose_stamped = PoseStamped()
        pose_stamped.header = header

        # Set end effector position
        pose_stamped.pose.position.x = goal_pos[0]
        pose_stamped.pose.position.y = goal_pos[1]
        pose_stamped.pose.position.z = goal_pos[2]
        
        # Set end effector quaternion
        pose_stamped.pose.orientation.x = 0
        pose_stamped.pose.orientation.y = 0.71
        pose_stamped.pose.orientation.z = 0
        pose_stamped.pose.orientation.w = 0.71

        # Add desired pose for inverse kinematics
        ik_request.pose_stamp.append(pose_stamped)
        # Request inverse kinematics from base to "right_hand" link
        ik_request.tip_names.append('right_hand')

        try:
            rospy.wait_for_service(service_name, 5.0)
            response = ik_service_proxy(ik_request)
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr("Service call failed: %s" % (e,))
            return

        # Check if result valid, and type of seed ultimately used to get solution
        if (response.result_type[0] > 0):
            # rospy.loginfo("SUCCESS!")
            # Format solution into Limb API-compatible dictionary
            return np.array(response.joints[0].position)
            
        return None



def main():
    """
    Main control loop to run the MPC controller and collect data for plotting.
    """
    rospy.init_node('mpc_sawyer_controller', anonymous=True)
    limb = Limb('right')  # Use 'right' for Sawyer's right arm
    rate = rospy.Rate(10)  # 10 Hz control loop
    delta_t = 0.1  # Time step

    tuck()

    # Subscribe to the goal_point topic to get the 3D goal position
    rospy.Subscriber("/strike_point", Point, goal_point_callback)
    
    while 'goal_position' not in globals():
        print("waiting for strike point...")

    # Convert goal position to goal joint angles with inverse kinematics
    global goal_position
    print("Goal Position: ", goal_position)
    target_pos = ik_service_client(goal_position)
    print(target_pos)

    # Initial and target positions
    current_position = np.array([-0.0003447265625, -0.9987998046875, -0.0013974609375,
                                  1.4995615234375, -0.0002099609375, -0.5000400390625,
                                  1.6983779296875])
    # target_position = np.array([-0.5, -1.5, -0.2, 1.5, -0.05, -1, 1])
    target_position = target_pos

    rospy.loginfo("Starting MPC control loop...")

    # Data collection
    time_data = []
    joint_states = {joint: [] for joint in limb.joint_names()}
    joint_velocities = {joint: [] for joint in limb.joint_names()}
    start_time = rospy.get_time()

    while not rospy.is_shutdown():
        # Get current joint angles from the robot
        current_state = np.array([limb.joint_angle(j) for j in limb.joint_names()])

        # Compute optimal joint velocities using MPC
        control_input = mpc_optimization(current_state, target_position)

        # Send the control input to the robot
        joint_command = dict(zip(limb.joint_names(), control_input))
        limb.set_joint_velocities(joint_command)

        # Collect data
        current_time = rospy.get_time() - start_time
        time_data.append(current_time)
        for idx, joint in enumerate(limb.joint_names()):
            joint_states[joint].append(current_state[idx])
            joint_velocities[joint].append(control_input[idx])

        # Debugging info
        rospy.loginfo(f"Current State: {current_state}")
        rospy.loginfo(f"Control Input: {control_input}")

        # Check if the robot has reached the target
        if np.linalg.norm(current_state - target_position) < 0.01:
            rospy.loginfo("Target reached!")
            break

        rate.sleep()

    # Plot data after movement
    plot_joint_data(time_data, joint_states, joint_velocities, limb.joint_names())


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("MPC control loop interrupted.")
