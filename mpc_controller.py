#!/usr/bin/env python

import rospy
import numpy as np
import cvxpy as cp
from intera_interface import Limb
import matplotlib.pyplot as plt


def mpc_optimization(current_position, target_position, horizon=20, delta_t=0.1):
    """
    Perform MPC to compute joint velocity commands for smooth movement with better handling for joint 6.
    """
    n_joints = len(current_position)

    # Decision variables
    U = cp.Variable((n_joints, horizon))  # Joint velocities
    X = cp.Variable((n_joints, horizon + 1))  # Predicted joint positions

    # Cost function
    cost = 0
    constraints = [X[:, 0] == current_position]

    for k in range(horizon):
        cost += cp.norm(X[:, k] - target_position, 2)  # Minimize tracking error
        cost += 0.2 * cp.norm(U[:, k], 2)  # Penalize control effort

        # Smoother velocity for joint 6
        if k < horizon - 1:
            cost += 0.05 * cp.norm(U[:, k + 1] - U[:, k], 2)  # Penalize velocity changes

        constraints += [X[:, k + 1] == X[:, k] + delta_t * U[:, k]]  # Dynamics
        constraints += [U[:, k] >= -0.7, U[:, k] <= 0.7]  # General velocity limits
        constraints += [U[6, k] >= -0.4, U[6, k] <= 0.4]  # Tighter limits for joint 6

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


def main():
    """
    Main control loop to run the MPC controller and collect data for plotting.
    """
    rospy.init_node('mpc_sawyer_controller', anonymous=True)
    limb = Limb('right')  # Use 'right' for Sawyer's right arm
    rate = rospy.Rate(10)  # 10 Hz control loop
    delta_t = 0.1  # Time step

    # Initial and target positions
    current_position = np.array([-0.0003447265625, -0.9987998046875, -0.0013974609375,
                                  1.4995615234375, -0.0002099609375, -0.5000400390625,
                                  1.6983779296875])
    target_position = np.array([-0.1, -1.5, -0.1,
                                  1.7, 0.0, -0.5,
                                  1.7])
    # target_position = np.array([-0.5, -1.5, -0.2, 1.5, -0.05, -1, 1])

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
