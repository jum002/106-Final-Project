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
import tf2_ros
from geometry_msgs.msg import Point, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA

from sklearn.cluster import DBSCAN, KMeans
from kalman_filter_pos import KalmanFilter3D


PLOTS_DIR = os.path.join(os.getcwd(), 'plots')

class ObjectDetector:
    def __init__(self):
        rospy.init_node('object_detector', anonymous=True)

        self.bridge = CvBridge()
        self.cv_color_image = None
        self.cv_depth_image = None

        self.color_image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.color_image_callback)
        self.depth_image_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_image_callback)
        self.camera_info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback)

        self.tf_listener = tf.TransformListener()  # Create a TransformListener object

        self.point_pub = rospy.Publisher("goal_point", Point, queue_size=10)
        self.point_stamped_pub = rospy.Publisher("goal_point_stamped", PointStamped, queue_size=10)
        self.image_pub = rospy.Publisher('detected_cup', Image, queue_size=10)
        self.marker_pub = rospy.Publisher('trajectory_marker_array', MarkerArray, queue_size=10)
        self.strike_zone_marker_pub = rospy.Publisher('strike_zone_marker', Marker, queue_size=10)
        # self.cam_ar_pub = rospy.Publisher('cam_ar_pos', Point, queue_size=10)

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
        self.launch_finished = False
        self.threshold = 5.0  # Threshold for launch detection
        self.g = 9.81  # Gravitational acceleration (m/s^2)
        self.predict_duration = 0.8 # in seconds
        self.t_plot = 0
        self.dt_frame = 1   # 0.1 seconds

        self.vel_depth_start = 2.2    # depth plane to start compute velocity
        self.vel_depth_end = 1.8    # depth plane to start compute velocity
        self.pos_depth_start = None
        self.pos_depth_end = None
        self.depth_t_start = None
        self.depth_t_end = None

        self.strike_rx = [0.2, 0.8]
        self.strike_ry = [-0.6, 0.6]
        self.strike_rz = [0, 1]
        self.strike_point = None
        self.strike_point_pub = rospy.Publisher("strike_point", Point, queue_size=10)
        self.strike_point_stamp_pub = rospy.Publisher('strike_point_stamped', PointStamped, queue_size=10)

        self.kalman_filter = KalmanFilter3D(process_noise=1e-2, measurement_noise=1e-1)

        # Get cam position from AR tag
        self.point_cam = None

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
            self.current_velocity = (self.curr_position - self.prev_position) / (self.dt_frame / self.FPS)
            self.prev_position = self.curr_position

    def process_images(self):
        hsv = cv2.cvtColor(self.cv_color_image, cv2.COLOR_BGR2HSV)

        # light green
        # lower_hsv = np.array([50, 80, 150])
        # upper_hsv = np.array([80, 255, 255])

        # light green (big)
        # lower_hsv = np.array([54, 125, 0])
        # upper_hsv = np.array([84, 255, 255])

        # blue
        # lower_hsv = np.array([98, 230, 60])
        # upper_hsv = np.array([108, 255, 210])

        # tennis
        lower_hsv = np.array([30, 125, 99])
        upper_hsv = np.array([55, 255, 255])

        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask_depth = np.zeros(self.cv_depth_image.shape)
        mask_depth[self.cv_depth_image < 3000] = 1.0

        y_coords, x_coords = np.nonzero(mask * mask_depth)

        if len(x_coords) == 0 or len(y_coords) == 0:
            return

        # K Mean clustering
        # coordinates = np.column_stack((x_coords, y_coords))
        # kmeans = KMeans(n_clusters=1)  # Adjust n_clusstriketers as needed
        # kmeans.fit(coordinates)
        # labels = kmeans.labels_
        # unique, counts = np.unique(labels, return_counts=True)
        # if len(unique) >= 1:  # More than just the noise (0 to n_clusters-1)
        #     main_cluster_label = unique[np.argmax(counts)]  # Find the largest cluster
        #     main_cluster_coords = coordinates[labels == main_cluster_label]
        #     print(len(x_coords))
        #     x_coords, y_coords = main_cluster_coords[:, 0], main_cluster_coords[:, 1]
        #     print("here: ", len(x_coords))

        coordinates = np.array(list(zip(x_coords, y_coords)))
        dbscan = DBSCAN(eps=3, min_samples=2)  # eps is the maximum distance between two samples to be considered as in the same neighborhood
        dbscan.fit(coordinates)
        labels = dbscan.labels_
        unique_labels = set(labels)
        if len(unique_labels - {-1}) >= 1:
            largest_cluster_label = max(unique_labels - {-1}, key=lambda label: list(labels).count(label))
            largest_cluster_points = coordinates[labels == largest_cluster_label]
            x_coords, y_coords = largest_cluster_points[:, 0], largest_cluster_points[:, 1]

        # compute 3D position of objectcurrent_position - self.last_position
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))  
        depth = self.cv_depth_image[center_y, center_x]

        camera_x, camera_y, camera_z = self.pixel_to_point(center_x, center_y, depth)
        camera_link_x, camera_link_y, camera_link_z = camera_z, -camera_x, -camera_y

        # Convert from mm to m
        camera_link_x /= 1000
        camera_link_y /= 1000
        camera_link_z /= 1000

        # try to get AR cam pos
        # if self.point_cam is None:
        #     self.point_cam = self.lookup_tag() self.predict_trajectory()
        # cam_offset = [0, 0, 0]
        # if self.point_cam is not None:
        #     cam_offset = [self.point_cam.x, self.point_cam.y, self.point_cam.z]
        # print(self.lookup_tag())
        cam_offset = [1.01, -0.343, 0.623]

        # Convert the (X, Y, Z) coordinates from camer self.predict_trajectory()
        try:
            self.tf_listener.waitForTransform("/base", "/camera_link", rospy.Time(), rospy.Duration(10.0))
            point_base = self.tf_listener.transformPoint("/base", PointStamped(header=Header(stamp=rospy.Time(), frame_id="camera_link"), point=Point(camera_link_x, camera_link_y, camera_link_z)))

            # Publish the transformed point
            noisy_position = np.array([point_base.point.x + cam_offset[0], point_base.point.y + cam_offset[1], point_base.point.z + cam_offset[2]])
            self.kalman_filter.predict()
            self.kalman_filter.update(noisy_position)
            smoothed_position = self.kalman_filter.get_position()

            point = Point(noisy_position[0], noisy_position[1], noisy_position[2])
            self.point_pub.publish(point)

            point_stamped = PointStamped()
            point_stamped.header.stamp = rospy.Time.now()  # Get the current timestamp
            point_stamped.header.frame_id = "base"  # Set the frame of reference
            point_stamped.point = point  # Assign the Point to the PointStamped
            self.point_stamped_pub.publish(point_stamped)

            # Overlay cup points on color image for viand np.abs(np.sum(self.curr_position)) > 0.1sualization
            cup_img = self.cv_color_image.copy()
            cup_img[y_coords, x_coords] = [0, 0, 255]  # Highlight cup points in red
            cv2.circle(cup_img, (center_x, center_y), 5, [0, 255, 0], -1)  # Draw green circle at center
            
            # Convert to ROS Image message and publish
            ros_image = self.bridge.cv2_to_imgmsg(cup_img, "bgr8")
            self.image_pub.publish(ros_image)

            # ==============================
            #  Trajectory Prediction
            #==============================

            # get position           
            current_position = np.array([point.x, point.y, point.z])
            current_depth = depth/1000
            # print(f"depth: {current_depth}")

            self.curr_position = current_position
            if self.prev_position is None:
                self.prev_position = current_position
            if np.linalg.norm(self.curr_position) < 0.1 and self.prev_position is not None:
                self.current_position = self.prev_position

            # compute current velocity
            self.update_current_velocity() 
            # print(np.linalg.norm(self.current_velocity))

            # detect start trajectory prediction (if velocity is greater than certain threshold)
            if self.vel_depth_end < current_depth <= self.vel_depth_start and not self.launch_detected and not self.launch_finished:
                self.launch_detected = True
                print("Launch detected! Depth = ", current_depth)
                self.predicted_trajectory.append(self.curr_position)
                self.pos_depth_start = self.curr_position
                self.depth_t_start = round(time.time() * 1000)

            if current_depth <= self.vel_depth_end and current_depth > 0.1 and self.launch_detected and not self.launch_finished:
                print("Launch detected again! Depth = ", current_depth)
                self.launch_finished = True
                self.predicted_trajectory.append(self.curr_position)
                self.pos_depth_end = self.curr_position
                self.depth_t_end = round(time.time() * 1000)
                self.predict_trajectory()
                self.strike_point = self.predict_strike_point_from_trajectory()
            
            if self.strike_point is not None:
                point_strike = Point(self.strike_point[0], self.strike_point[1], self.strike_point[2])
                self.strike_point_pub.publish(point_strike)
                point_strike_stamped = PointStamped()
                point_strike_stamped.header.stamp = rospy.Time.now()  # Get the current timestamp
                point_strike_stamped.header.frame_id = "base"  # Set the frame of reference
                point_strike_stamped.point = point_strike  # Assign the Point to the PointStamped
                self.strike_point_stamp_pub.publish(point_strike_stamped)

            self.publish_traj()
            self.publish_strike_zone()

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            print("TF Error: " + e)
            return
        
    def predict_trajectory(self):
        dt = self.dt_frame / self.FPS  # Time interval betself.pos_depth_start, ween frames (30 FPS)
        t = 0  # Start time
        print("Predicting trajectory: ", self.pos_depth_start, self.pos_depth_end, self.depth_t_start, self.depth_t_end)
        vel = (self.pos_depth_end - self.pos_depth_start) / ((self.depth_t_end - self.depth_t_start) / 1000)
        print("Predict vel: ", vel)
        while t < self.predict_duration:
            # 3D projectile motion equations
            x = self.pos_depth_end[0] + vel[0] * t
            y = self.pos_depth_end[1] + vel[1] * t
            z = self.pos_depth_end[2] + vel[2] * t - 0.5 * self.g * t**2
            self.predicted_trajectory.append((x, y, z))
            t += dt  # Increment time

    def predict_strike_point_from_trajectory(self):
        for pt in self.predicted_trajectory:
            if self.strike_rx[0] <= pt[0] <= self.strike_rx[1] and \
                self.strike_ry[0] <= pt[1] <= self.strike_ry[1] and \
                self.strike_rz[0] <= pt[2] <= self.strike_rz[1]:
                return pt
        return None

    def publish_traj(self):
        trajectory = [Point(p[0], p[1], p[2]) for p in self.predicted_trajectory]
        
        # Create a MarkerArray to hold the markers
        marker_array = MarkerArray()
        
        # Add markers (points or spheres) to represent the trajectory
        for i, pt in enumerate(trajectory):
            marker = Marker()
            marker.header.frame_id = "base"  # Use appropriate frame (like "map" or "base_link")
            marker.header.stamp = rospy.Time.now()
            marker.ns = "trajectory"
            marker.id = i
            marker.type = Marker.SPHERE  # Use SPHERE for points
            marker.action = Marker.ADD
            marker.pose.position = pt
            marker.scale.x = 0.1  # Sphere size (radius)
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color = ColorRGBA(0.0, 1.0, 0.0, 1.0)  # Green color
            
            # Append marker to the MarkerArray
            marker_array.markers.append(marker)
        
        # Publish the MarkerArray
        self.marker_pub.publish(marker_array)

    def publish_strike_zone(self):
        # Create the Marker object
        marker = Marker()
        marker.header.frame_id = "base"  # Reference frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "transparent_box"
        marker.id = 0
        marker.type = Marker.CUBE  # Type is CUBE for a rectangular box
        marker.action = Marker.ADD

        # Set the position of the box
        marker.pose.position.x = (self.strike_rx[1] + self.strike_rx[0]) / 2
        marker.pose.position.y = (self.strike_ry[1] + self.strike_ry[0]) / 2
        marker.pose.position.z = (self.strike_rz[1] + self.strike_rz[0]) / 2

        # Set the orientation (no rotation)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # Set the scale (dimensions of the box)
        marker.scale.x = self.strike_rx[1] - self.strike_rx[0]  # Length
        marker.scale.y = self.strike_ry[1] - self.strike_ry[0]
        marker.scale.z = self.strike_rz[1] - self.strike_rz[0]  # Height

        # Set the color with transparency (alpha = 0.2 for 80% transparency)
        marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.2)  # Blue with transparency

        self.strike_zone_marker_pub.publish(marker)

    def lookup_tag(self, tag_number=0):
        """
        Given an AR tag number, this returns the position of the AR tag in the robot's base frame.
        You can use either this function or try starting the scripts/tag_pub.py script.  More info
        about that script is in that file.  

        Parameters
        ----------
        tag_number : int

        Returns
        -------
        3x' :obj:`numpy.ndarray`
            tag position
        """
        # TODO: initialize a tf buffer and listener as in lab 4
        tfBuffer = tf2_ros.Buffer()
        tfListener = tf2_ros.TransformListener(tfBuffer)

        try:
            trans = tfBuffer.lookup_transform('base', f'ar_marker_{tag_number}', rospy.Time(0), rospy.Duration(10.0))
        except Exception as e:
            print(e)
            print("Retrying ...")

        tag_pos = [getattr(trans.transform.translation, dim) for dim in ('x', 'y', 'z')]
        return Point(tag_pos[0], tag_pos[1], tag_pos[2])

if __name__ == '__main__':
    ObjectDetector()
