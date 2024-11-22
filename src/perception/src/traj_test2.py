#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tf
from geometry_msgs.msg import Point

class ObjectDetector:
    def __init__(self):
        rospy.init_node('object_detector', anonymous=True)

        self.bridge = CvBridge()
        self.cv_color_image = None
        self.cv_depth_image = None

        self.color_image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.color_image_callback)
        self.depth_image_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_image_callback)
        self.camera_info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback)

        self.point_pub = rospy.Publisher("goal_point", Point, queue_size=10)
        self.image_pub = rospy.Publisher('detected_cup', Image, queue_size=10)

        self.fx, self.fy, self.cx, self.cy = None, None, None, None
        self.last_position = None
        self.curr_position = None
        self.trajectory = []
        self.predicted_trajectory = []

        self.FPS = 30
        self.global_time = 0
        self.prev_position = None
        self.current_velocity = None
        self.velocities = []

        self.launch_detected = False
        self.threshold = 0.05  # Threshold for launch detection
        self.g = 9.81  # Gravitational acceleration (m/s^2)
        self.predict_duration = 0.5 # in seconds
        self.t_plot = 0
        self.dt_frame = 2

        rospy.spin()

    def camera_info_callback(self, msg):
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]      
        self.cy = msg.K[5]

    def pixel_to_point(self, u, v, depth):
        X = (u - self.cx) * depth / self.fx
        Y = (v - self.cy) * depth / self.fy
        Z = depth
        return X, Y, Z

    def color_image_callback(self, msg):
        try:
            self.cv_color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if self.cv_depth_image is not None:
                self.global_time += 1
                self.process_images()
        except Exception as e:
            print("Error:", e)

    def depth_image_callback(self, msg):
        try:
            self.cv_depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        except Exception as e:
            print("Error:", e)

    def update_current_velocity(self):
        assert(self.curr_position is not None and self.prev_position is not None)
        if self.global_time % self.dt_frame == 0:
            self.current_velocity = (self.curr_position - self.prev_position) * (self.dt_frame / self.FPS)
            self.prev_position = self.curr_position

    def process_images(self):
        hsv = cv2.cvtColor(self.cv_color_image, cv2.COLOR_BGR2HSV)

        # light green
        lower_hsv = np.array([50, 80, 150])
        upper_hsv = np.array([80, 255, 255])

        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask_depth = np.zeros(self.cv_depth_image.shape)
        mask_depth[self.cv_depth_image < 3000] = 1.0

        y_coords, x_coords = np.nonzero(mask * mask_depth)

        if len(x_coords) == 0 or len(y_coords) == 0:
            return

        # compute 3D position of objectcurrent_position - self.last_position
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))  
        depth = self.cv_depth_image[center_y, center_x]

        # camera_x, camera_y, camera_z = self.pixel_to_point(center_x, center_y, depth)
        # camera_link_x, camera_link_y, camera_link_z = camera_z / 1000, -camera_x / 1000, -camera_y / 1000

        camera_x, camera_y, camera_z = self.pixel_to_point(center_x, center_y, depth)
        camera_link_x, camera_link_y, camera_link_z = camera_z, -camera_x, -camera_y
        # Convert from mm to m
        camera_link_x /= 1000
        camera_link_y /= 1000
        camera_link_z /= 1000

        current_position = np.array([camera_link_x, camera_link_y, camera_link_z])
        # self.trajectory.append(current_position)
        print(current_position)
        # if self.global_time > 30 * 10:
        #     self.visualize_actual_trajectory()
        # return

        self.curr_position = current_position
        if self.prev_position is None:
            self.prev_position = current_position

        # compute current velocity
        self.update_current_velocity() 

        # detect start trajectory prediction (if velocity is greater than certain threshold)
        if not self.launch_detected and np.abs(np.sum(self.curr_position)) > 0.1 and self.current_velocity is not None and \
                np.linalg.norm(self.current_velocity) > self.threshold:
            self.launch_detected = True
            print("Launch detected!")
            self.predicted_trajectory.append(self.curr_position)
            self.velocities.append(self.current_velocity)
            self.predict_trajectory()

        if self.launch_detected and np.abs(np.sum(self.curr_position)) > 0.1: 
            self.trajectory.append(self.curr_position)
            self.t_plot += self.dt_frame / self.FPS

        if self.t_plot > self.predict_duration:
            print("here")
            self.visualize_trajectory()


    def predict_trajectory(self):
        dt = self.dt_frame / self.FPS  # Time interval between frames (30 FPS)
        t = 0  # Start time
        while t < self.predict_duration:
            # 3D projectile motion equations
            x = self.prev_position[0] + self.current_velocity[0] * t
            y = self.prev_position[1] + self.current_velocity[1] * t
            z = self.prev_position[2] + self.current_velocity[2] * t - 0.5 * self.g * t**2
            self.predicted_trajectory.append((x, y, z))
            t += dt  # Increment time

        # self.visualize_trajectory()

    def visualize_actual_trajectory(self):
        actual_x = [p[0] for p in self.trajectory] # rosrun perception traj_test2.py
        actual_y = [p[1] for p in self.trajectory]
        actual_z = [p[2] for p in self.trajectory]

        fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # ax.plot(actual_x, actual_y, actual_z, label="Actual Trajectory", marker='o')
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(actual_x, actual_y, actual_z)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("3D Ball Trajectory")
        ax.legend()

        plt.show()

    def visualize_trajectory(self):
        actual_x = [p[0] for p in self.trajectory] # rosrun perception traj_test2.py
        actual_y = [p[1] for p in self.trajectory]
        actual_z = [p[2] for p in self.trajectory]

        predicted_x = [p[0] for p in self.predicted_trajectory]
        predicted_y = [p[1] for p in self.predicted_trajectory]
        predicted_z = [p[2] for p in self.predicted_trajectory]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(actual_x, actual_y, actual_z, label="Actual Trajectory", marker='o')
        ax.plot(predicted_x, predicted_y, predicted_z, label="Predicted Trajectory", linestyle='--')

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("3D Ball Trajectory")
        ax.legend()

        plt.show()

if __name__ == '__main__':
    ObjectDetector()
