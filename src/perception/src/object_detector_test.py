#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo # For camera intrinsic parameters
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import os
import time
import tf
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import Header


PLOTS_DIR = os.path.join(os.getcwd(), 'plots')

class ObjectDetector:
    def __init__(self):
        rospy.init_node('object_detector', anonymous=True)

        self.bridge = CvBridge()

        self.cv_color_image = None
        self.cv_depth_image = None

        self.color_image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.color_image_callback)
        self.depth_image_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_image_callback)

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.camera_info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback)

        self.tf_listener = tf.TransformListener()  # Create a TransformListener object

        self.point_pub = rospy.Publisher("goal_point", Point, queue_size=10)
        self.image_pub = rospy.Publisher('detected_cup', Image, queue_size=10)

        rospy.spin()

    def camera_info_callback(self, msg):
        # TODO: Extract the intrinsic parameters from the CameraInfo message (look this message type up online)
        self.fx = msg.K[0] # 306.6052551269531
        self.fy = msg.K[4] # 306.8348388671875
        self.cx = msg.K[2] # 212.38922119140625
        self.cy = msg.K[5] # 120.1928482055664

    def pixel_to_point(self, u, v, depth):
        # TODO: Use the camera intrinsics to convert pixel coordinates to real-world coordinates
        X = (u - self.cx) * depth / self.fx
        Y = (v - self.cy) * depth / self.fy
        Z = depth
        return X, Y, Z

    def color_image_callback(self, msg):
        try:
            # Convert the ROS Image message to an OpenCV image (BGR8 format)
            self.cv_color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # If we have both color and depth images, process them
            if self.cv_depth_image is not None:
                self.process_images()

        except Exception as e:
            print("Error:", e)

    def depth_image_callback(self, msg):
        try:
            # Convert the ROS Image melower_hsv = np.array([87, 151, 36])
            self.cv_depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")

        except Exception as e:
            print("Error:", e)

    def process_images(self):
        # Convert the color image to HSV color space
        hsv = cv2.cvtColor(self.cv_color_image, cv2.COLOR_BGR2HSV)
        # TODO: Define range for cup color in HSVdepth_width:=424 depth_height:=240
        # Run `python hsv_color_thresholder.py` and tune the bounds so you only see your cup
        # update lower_hsv and upper_hsv directly

        # lower_hsv = np.array([35, 51, 69]) # TODO: Define lower HSV values for cup color
        # upper_hsv = np.array([88, 255, 223]) # TODO: Define upper HSV values for cup color

        # light green
        lower_hsv = np.array([50, 80, 150])
        upper_hsv = np.array([80, 255, 255])

        # dark green
        # lower_hsv = np.array([87, 151, 36])
        # upper_hsv = np.array([93, 255, 255])

        # lower_hsv = np.array([0, 91, 43]) # TODO: Define lower HSV values for cup color
        # upper_hsv = np.array([5, 255, 255])        

        # TODO: Threshold the image to get only cup colors
        # HINT: Lookup cv2.inRange()
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask_depth = np.zeros(self.cv_depth_image.shape)
        mask_depth[self.cv_depth_image < 3000] = 1.0
        # plt.imshow(mask_depth, cmap='gray')
        # plt.show()

        # TODO: Get the coordinates of the cup points on the mask
        # HINT: Lookup np.nonzero()
        y_coords, x_coords = np.nonzero(mask * mask_depth)

        # If there are no detected points, exit
        if len(x_coords) == 0 or len(y_coords) == 0:
            print("No points detected/home/cc/ee106a/fa24/class/ee106a-agn/ros_workspaces/lab8/src. Is your color filter wrong?")
            return

        # Calculate the center of the detected region by 
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))

        # Fetch the depth value at the center
        depth = self.cv_depth_image[center_y, center_x]

        if self.fx and self.fy and self.cx and self.cy:
            camera_x, camera_y, camera_z = self.pixel_to_point(center_x, center_y, depth)
            camera_link_x, camera_link_y, camera_link_z = camera_z, -camera_x, -camera_y
            # Convert from mm to m
            camera_link_x /= 1000
            camera_link_y /= 1000
            camera_link_z /= 1000

            # Convert the (X, Y, Z) coordinates from camera frame to odom frame
            try:
                print("Publishing goal point: ", camera_link_x, camera_link_y, camera_link_z)
                # Publish the transformed point
                self.point_pub.publish(Point(camera_link_x, camera_link_y, camera_link_z))

                # Overlay cup points on color image for visualization
                cup_img = self.cv_color_image.copy()
                cup_img[y_coords, x_coords] = [0, 0, 255]  # Highlight cup points in red
                cv2.circle(cup_img, (center_x, center_y), 5, [0, 255, 0], -1)  # Draw green circle at center
                
                # Convert to ROS Image message and publish
                ros_image = self.bridge.cv2_to_imgmsg(cup_img, "bgr8")
                self.image_pub.publish(ros_image)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                print("TF Error: " + e)
                return

if __name__ == '__main__':
    ObjectDetector()
