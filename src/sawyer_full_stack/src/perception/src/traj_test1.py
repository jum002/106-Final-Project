#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time

class BallTrajectoryPredictor:
    def __init__(self):
        rospy.init_node('ball_trajectory_predictor', anonymous=True)

        self.bridge = CvBridge()

        # 3D points of the ball (X, Y, Z)
        self.actual_trajectory = []
        self.timestamps = []

        # Initial velocities and gravity
        self.initial_velocity = None
        self.gravity = 9.81  # m/s^2

        # ROS Subscribers and Publishers
        self.color_image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.color_image_callback)
        self.depth_image_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_image_callback)
        self.camera_info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback)
        self.point_pub = rospy.Publisher("goal_point", Point, queue_size=10)
        self.image_pub = rospy.Publisher('detected_cup', Image, queue_size=10)

        # Camera intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # To store the current images
        self.cv_color_image = None
        self.cv_depth_image = None

        # Launch detection
        self.launch_detected = False
        self.launch_threshold = 0.1  # Minimum displacement in meters to consider the ball launched

        rospy.spin()

    def camera_info_callback(self, msg):
        """Callback to retrieve intrinsic parameters from the camera info."""
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]

    def pixel_to_point(self, u, v, depth):
        """Convert pixel coordinates (u, v) and depth to 3D camera coordinates."""
        X = (u - self.cx) * depth / self.fx
        Y = (v - self.cy) * depth / self.fy
        Z = depth
        return X, Y, Z

    def color_image_callback(self, msg):
        """Callback to handle color image processing."""
        try:
            self.cv_color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if self.cv_depth_image is not None:
                self.process_images()
        except Exception as e:
            rospy.logerr(f"Color image callback error: {e}")

    def depth_image_callback(self, msg):
        """Callback to handle depth image processing."""
        try:
            self.cv_depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        except Exception as e:
            rospy.logerr(f"Depth image callback error: {e}")

    def process_images(self):
        """Process the color and depth images to detect the ball and predict trajectory."""
        if not all([self.fx, self.fy, self.cx, self.cy]):
            rospy.logwarn("Camera intrinsics not yet available. Skipping image processing.")
            return

        hsv = cv2.cvtColor(self.cv_color_image, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([35, 51, 69]) # TODO: Define lower HSV values for cup color
        upper_hsv = np.array([88, 255, 223]) # TODO: Define upper HSV values for cup color
        # lower_hsv = np.array([0, 91, 43])
        # upper_hsv = np.array([5, 255, 255])

        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        y_coords, x_coords = np.nonzero(mask)

        if len(x_coords) == 0 or len(y_coords) == 0:
            rospy.loginfo("No points derospy.loginfotected. Check your HSV filter.")
            return

        # Calculate the center of the detected region by 
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))

        # Fetch the depth value at the center
        depth = self.cv_depth_image[center_y, center_x]

        camera_x, camera_y, camera_z = self.pixel_to_point(center_x, center_y, depth)
        camera_link_x, camera_link_y, camera_link_z = camera_z, -camera_x, -camera_y
        # Convert from mm to m
        camera_link_x /= 1000
        camera_link_y /= 1000
        camera_link_z /= 1000

        camera_x, camera_y, camera_z = camera_link_x, camera_link_y, camera_link_z
        current_time = rospy.get_time()

        # Append the point and timestamrospy.loginfop
        self.actual_trajectory.append([camera_x, camera_y, camera_z])
        self.timestamps.append(current_time)

        # Publish the transformed point
        self.point_pub.publish(Point(camera_x, camera_y, camera_z))

        # Overlay cup points on color image for visualization
        cup_img = self.cv_color_image.copy()
        cup_img[y_coords, x_coords] = [0, 0, 255]  # Highlight cup points in red
        cv2.circle(cup_img, (center_x, center_y), 5, [0, 255, 0], -1)  # Draw green circle at center
        
        # Convert to ROS Image message and publish
        ros_image = self.bridge.cv2_to_imgmsg(cup_img, "bgr8")
        self.image_pub.publish(ros_image)

        if len(self.actual_trajectory) > 1:
            last_point = self.actual_trajectory[-2]
            displacement = np.sqrt((camera_x - last_point[0])**2 +
                                   (camera_y - last_point[1])**2 +
                                   (camera_z - last_point[2])**2)

            # Detect launch based on displacement
            if not self.launch_detected and displacement > self.launch_threshold:
                self.launch_detected = True
                print("Ball launch detected!")

            # Calculate initial velocity after launch
            if self.launch_detected and len(self.actual_trajectory) > 2:
                last_time = self.timestamps[-2]
                vx = (camera_x - last_point[0]) / (current_time - last_time)
                vy = (camera_y - last_point[1]) / (current_time - last_time)
                vz = (camera_z - last_point[2]) / (current_time - last_time)

                self.initial_velocity = [vx, vy, vz]
                print(f"Estimated initial velocity: ({vx:.2f}, {vy:.2f}, {vz:.2f})")

        # Publish the detected point
        self.point_pub.publish(Point(camera_x, camera_y, camera_z))

        # Predict and visualize trajectory if velocities are available
        if self.launch_detected and self.initial_velocity is not None:
            self.predict_and_plot_trajectory()

    def predict_and_plot_trajectory(self):
        """Predict the ball's trajectory and plot it against the actual trajectory."""
        if not self.initial_velocity:
            return

        # Initial position and velocity
        x0, y0, z0 = self.actual_trajectory[0]
        vx, vy, vz = self.initial_velocity

        # Simulate the predicted trajectory
        t_pred = np.linspace(0, 2, 100)  # Simulate for 2 seconds
        x_pred = x0 + vx * t_pred
        y_pred = y0 + vy * t_pred
        z_pred = z0 + vz * t_pred - 0.5 * self.gravity * t_pred**2

        # Convert actual trajectory to numpy arrays
        actual_points = np.array(self.actual_trajectory)
        actual_times = np.array(self.timestamps) - self.timestamps[0]

        # Plot the actual and predicted trajectories
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            actual_points[:, 0], actual_points[:, 1], actual_points[:, 2], 
            c=actual_times, cmap='viridis', label="Actual Trajectory"
        )
        cbar = fig.colorbar(scatter, ax=ax, label='Time (s)')

        ax.plot(x_pred, y_pred, z_pred, label="Predicted Trajectory", c='b')
        for i in range(0, len(t_pred), 10):
            ax.text(x_pred[i], y_pred[i], z_pred[i], f"{t_pred[i]:.1f}s", color='red')

        ax.set_xlabel("X (Depth)")
        ax.set_ylabel("Y (Left)")
        ax.set_zlabel("Z (Up)")
        ax.legend()
        plt.show()

if __name__ == '__main__':
    BallTrajectoryPredictor()
