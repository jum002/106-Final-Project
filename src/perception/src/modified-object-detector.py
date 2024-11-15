#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import os
import time
import tf
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import Header
from scipy.optimize import curve_fit  # Add this import

PLOTS_DIR = os.path.join(os.getcwd(), 'plots')
if not os.path.exists(PLOTS_DIR):  # Add this check
    os.makedirs(PLOTS_DIR)

class ObjectDetector:
    def __init__(self):
        rospy.init_node('object_detector', anonymous=True)

        self.bridge = CvBridge()

        self.cv_color_image = None
        self.cv_depth_image = None

        # Add trajectory tracking
        self.trajectory_points = {'x': [], 'y': [], 'z': [], 'time': []}
        self.start_time = time.time()
        
        # Add prediction parameters
        self.prediction_horizon = 1.0  # Predict 1 second into future
        self.n_prediction_points = 20  # Number of points in prediction
        self.min_points_for_prediction = 5  # Minimum points needed for prediction

        self.color_image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.color_image_callback)
        self.depth_image_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_image_callback)

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.camera_info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback)

        self.tf_listener = tf.TransformListener()

        self.point_pub = rospy.Publisher("goal_point", Point, queue_size=10)
        self.image_pub = rospy.Publisher('detected_cup', Image, queue_size=10)

        # Add timer for trajectory plotting
        self.plot_timer = rospy.Timer(rospy.Duration(5.0), self.plot_trajectory)

        rospy.spin()

    # Keep your existing callbacks the same
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
                self.process_images()
        except Exception as e:
            print("Error:", e)

    def depth_image_callback(self, msg):
        try:
            self.cv_depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        except Exception as e:
            print("Error:", e)

    # Add new method for trajectory prediction
    def predict_trajectory(self):
        """Predict future trajectory based on current observations"""
        if len(self.trajectory_points['time']) < self.min_points_for_prediction:
            return None, None, None
        
        # Convert lists to numpy arrays
        t = np.array(self.trajectory_points['time'])
        x = np.array(self.trajectory_points['x'])
        y = np.array(self.trajectory_points['y'])
        z = np.array(self.trajectory_points['z'])
        
        # Create time points for prediction
        last_time = t[-1]
        future_times = np.linspace(last_time, 
                                 last_time + self.prediction_horizon, 
                                 self.n_prediction_points)

        try:
            # Fit quadratic functions for each coordinate
            def parabola(t, a, b, c):
                return a * t**2 + b * t + c

            # Fit parameters for each dimension
            px, _ = curve_fit(parabola, t, x)
            py, _ = curve_fit(parabola, t, y)
            pz, _ = curve_fit(parabola, t, z)

            # Generate predictions
            x_pred = parabola(future_times, *px)
            y_pred = parabola(future_times, *py)
            z_pred = parabola(future_times, *pz)

            return x_pred, y_pred, z_pred

        except Exception as e:
            print(f"Prediction error: {e}")
            return None, None, None

    # Add new method for trajectory plotting
    def plot_trajectory(self, event=None):
        if len(self.trajectory_points['x']) < 2:
            return

        # Create 3D trajectory plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot actual trajectory
        ax.plot(self.trajectory_points['x'], 
                self.trajectory_points['y'], 
                self.trajectory_points['z'], 
                'b-', label='Actual Trajectory', linewidth=2)
        
        # Plot predicted trajectory
        x_pred, y_pred, z_pred = self.predict_trajectory()
        if x_pred is not None:
            ax.plot(x_pred, y_pred, z_pred, 'r--', 
                   label='Predicted Trajectory', linewidth=2)
        
        # Add points
        ax.scatter(self.trajectory_points['x'][0], 
                  self.trajectory_points['y'][0], 
                  self.trajectory_points['z'][0], 
                  color='green', s=100, label='Start')
        ax.scatter(self.trajectory_points['x'][-1], 
                  self.trajectory_points['y'][-1], 
                  self.trajectory_points['z'][-1], 
                  color='blue', s=100, label='Current')

        if x_pred is not None:
            ax.scatter(x_pred[-1], y_pred[-1], z_pred[-1], 
                      color='red', s=100, label='Predicted End')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Object Trajectory with Prediction')
        ax.legend()
        ax.grid(True)
        ax.view_init(elev=20, azim=45)

        # Save the plot
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig(os.path.join(PLOTS_DIR, f'trajectory_{timestamp}.png'))
        plt.close()

    def process_images(self):
        hsv = cv2.cvtColor(self.cv_color_image, cv2.COLOR_BGR2HSV)
        
        lower_hsv = np.array([0, 91, 43])
        upper_hsv = np.array([5, 255, 255])
        
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        y_coords, x_coords = np.nonzero(mask)

        if len(x_coords) == 0 or len(y_coords) == 0:
            print("No points detected. Is your color filter wrong?")
            return

        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        depth = self.cv_depth_image[center_y, center_x]

        if self.fx and self.fy and self.cx and self.cy:
            camera_x, camera_y, camera_z = self.pixel_to_point(center_x, center_y, depth)
            camera_link_x, camera_link_y, camera_link_z = camera_z, -camera_x, -camera_y
            # Convert from mm to m
            camera_link_x /= 1000
            camera_link_y /= 1000
            camera_link_z /= 1000

            try:
                print("Publishing goal point: ", camera_link_x, camera_link_y, camera_link_z)
                
                # Store trajectory point
                self.trajectory_points['x'].append(camera_link_x)
                self.trajectory_points['y'].append(camera_link_y)
                self.trajectory_points['z'].append(camera_link_z)
                self.trajectory_points['time'].append(time.time() - self.start_time)
                
                # Publish the point
                self.point_pub.publish(Point(camera_link_x, camera_link_y, camera_link_z))

                # Visualize on image
                cup_img = self.cv_color_image.copy()
                cup_img[y_coords, x_coords] = [0, 0, 255]
                cv2.circle(cup_img, (center_x, center_y), 5, [0, 255, 0], -1)
                
                ros_image = self.bridge.cv2_to_imgmsg(cup_img, "bgr8")
                self.image_pub.publish(ros_image)
                
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                print("TF Error: ", e)
                return

if __name__ == '__main__':
    ObjectDetector()
