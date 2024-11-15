#!/usr/bin/env python
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

cv2.namedWindow('image')
cv2.createTrackbar('HMin', 'image', 0, 179, lambda x: None)  # Hue is from 0-179 for OpenCV
cv2.createTrackbar('SMin', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('VMin', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('HMax', 'image', 0, 179, lambda x: None)
cv2.createTrackbar('SMax', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('VMax', 'image', 0, 255, lambda x: None)

# Set default value for MAX HSV trackbars.
cv2.setTrackbarPos('HMax', 'image', 179)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)

# Initialize variables
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

# subscribes to the image topic that has the rgb images from the realsense
class ImageSubscriber:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.img = None

    def image_callback(self, msg):
        try:
            # Convert the ROS image to OpenCV format
            self.img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

    def process_image(self):
        if self.img is None:
            return
        # Get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin', 'image')
        sMin = cv2.getTrackbarPos('SMin', 'image')
        vMin = cv2.getTrackbarPos('VMin', 'image')
        hMax = cv2.getTrackbarPo lower_hsv = np.array([35, 51, 69]) # TODO: Define lower HSV values for cup color
        # upper_hsv = np.array([88, 255, 223])s('HMax', 'image')
        sMax = cv2.getTrackbarPos('SMax', 'image')
        vMax = cv2.getTrackbarPos('VMax', 'image')
        # Set minimum and maximum HSV values for thresholding
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])
        # Convert the image to HSV and apply the threshold
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(self.img, self.img, mask=mask)

        cv2.imshow('thresholded_img', output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown('User requested shutdown.')

def main():
    # Initialize the ROS node
    rospy.init_node('hsv_color_thresholder', anonymous=True)
    # Create our subscriber
    img_subscriber = ImageSubscriber()

    # Main loop keeps updating the namedwindow
    while not rospy.is_shutdown():
        img_subscriber.process_image()

    # Close all OpenCV windows after exiting
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
