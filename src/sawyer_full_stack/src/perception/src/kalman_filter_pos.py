import numpy as np

class KalmanFilter3D:
    def __init__(self, process_noise=1e-2, measurement_noise=1e-1):
        # State vector [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)

        # State transition matrix (constant velocity model)
        self.F = np.eye(6)
        for i in range(3):
            self.F[i, i+3] = 1  # Position depends on velocity

        # Process noise covariance
        self.Q = np.eye(6) * process_noise

        # Measurement matrix (we measure position only)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1

        # Measurement noise covariance
        self.R = np.eye(3) * measurement_noise

        # State covariance
        self.P = np.eye(6)

    def predict(self):
        # Predict state and covariance
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, measurement):
        # Measurement residual
        y = measurement - np.dot(self.H, self.state)

        # Residual covariance
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Kalman gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Update state
        self.state = self.state + np.dot(K, y)

        # Update covariance
        self.P = np.dot(np.eye(len(self.P)) - np.dot(K, self.H), self.P)

    def get_position(self):
        return self.state[:3]  # Return the smoothed position


class ObjectDetector:
    def __init__(self):
        # ... (other initializations)
        self.kalman_filter = KalmanFilter3D(process_noise=1e-2, measurement_noise=1e-1)

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

        # Compute 3D position of the object
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        depth = self.cv_depth_image[center_y, center_x]

        camera_x, camera_y, camera_z = self.pixel_to_point(center_x, center_y, depth)
        camera_link_x, camera_link_y, camera_link_z = camera_z / 1000, -camera_x / 1000, -camera_y / 1000

        noisy_position = np.array([camera_link_x, camera_link_y, camera_link_z])

        # Kalman filter processing
        self.kalman_filter.predict()
        self.kalman_filter.update(noisy_position)

        smoothed_position = self.kalman_filter.get_position()
        print("Smoothed Position:", smoothed_position)

        # Publish smoothed position
        goal_point = Point()
        goal_point.x, goal_point.y, goal_point.z = smoothed_position
        self.point_pub.publish(goal_point)

        # Continue with existing logic
        self.curr_position = smoothed_position
