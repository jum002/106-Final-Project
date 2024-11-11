#!/usr/bin/env python
#The line above tells Linux that this file is a Python script,
#and that the OS should use the Python interpreter in /usr/bin/env
#to run it. Don't forget to use "chmod +x [filename]" to make
#this script executable.

#Import the rospy package. For an import to work, it must be specified
#in both the package manifest AND the Python file in which it is used.
import rospy
import tf2_ros
import tf
import sys
import numpy as np
from geometry_msgs.msg import TransformStamped, PoseStamped, Twist, Point
from tf.transformations import quaternion_from_euler
from tf2_geometry_msgs import do_transform_pose
from trajectory import plan_curved_trajectory

#Define the method which contains the main functionality of the node.
def controller(waypoint):
  """
  Controls a turtlebot whose position is denoted by turtlebot_frame,
  to go to a position denoted by target_frame
  Inputs:
  - turtlebot_frame: the tf frame of the AR tag on your turtlebot
  - goal_frame: the tf frame of the target AR tag
  """

  # Create a publisher and a tf buffer, which is primed with a tf listener
  pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
  tfBuffer = tf2_ros.Buffer()
  tfListener = tf2_ros.TransformListener(tfBuffer)

  # Create a timer object that will sleep long enough to result in
  # a 10Hz publishing rate
  r = rospy.Rate(10) # 10hz
  # you can also use the rate to calculate your dt, but you don't have to

  # All in the form [x, y]
  # NOTE: The Turtlebot typically does not need an integral term so we set it to 0 to make this a PD controller
  Kp = np.diag([2, 0.8]) # TODO: You may need to tune these values for your turtlebot
  Kd = np.diag([-0.5, 0.5]) # TODO: You may need to tune these values for your turtlebot
  Ki = np.diag([0, 0])

  prev_time = rospy.get_time()
  integ = np.zeros(2)
  derivative = np.zeros(2)
  previous_error = np.zeros(2)

  # Loop until the node is killed with Ctrl-C
  while not rospy.is_shutdown():
    try:
      #                                              target_frame, source_frame, current_time_in_ros, how long to wait for transform lookup
      trans_odom_to_base_link = tfBuffer.lookup_transform('base_footprint', 'odom', rospy.Time(), rospy.Duration(5)) # TODO: create a transform between odom to base link

      (roll, pitch, baselink_yaw) = tf.transformations.euler_from_quaternion(
        [trans_odom_to_base_link.transform.rotation.x, trans_odom_to_base_link.transform.rotation.y,
            trans_odom_to_base_link.transform.rotation.z, trans_odom_to_base_link.transform.rotation.w])


      waypoint_trans = PoseStamped() 
      waypoint_trans.pose.position.x = waypoint[0]
      waypoint_trans.pose.position.y = waypoint[1]
      waypoint_trans.pose.position.z = 0 # Assuming the waypoint is on the ground

      quat = quaternion_from_euler(0, 0, waypoint[2])
      waypoint_trans.pose.orientation.x = quat[0]
      waypoint_trans.pose.orientation.y = quat[1] 
      waypoint_trans.pose.orientation.z = quat[2] 
      waypoint_trans.pose.orientation.w = quat[3] 

      # Use the transform to compute the waypoint's pose in the base_link frame
      waypoint_in_base_link = do_transform_pose(waypoint_trans, trans_odom_to_base_link) 
      (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
        [waypoint_in_base_link.pose.orientation.x, waypoint_in_base_link.pose.orientation.y,
            waypoint_in_base_link.pose.orientation.z, waypoint_in_base_link.pose.orientation.w])


      curr_time = rospy.get_time()

      # some debug output below
      print(f"Current: {trans_odom_to_base_link.transform.translation.x}, {trans_odom_to_base_link.transform.translation.y}, {baselink_yaw  }")
      print(f"Target: {waypoint}")

      # Process trans to get your state error
      # Generate a control command to send to the robot
      x_error = waypoint_in_base_link.pose.position.x
      y_error = waypoint_in_base_link.pose.position.y
      error = np.array([x_error, yaw])
      
      # proportional term
      proportional = np.dot(Kp, error).squeeze()
      
      # integral term
      dt = curr_time - prev_time
      integ += error * dt
      integral = np.dot(Ki, integ).squeeze()

      # dervative term
      error_deriv = (error - previous_error) / dt
      derivative = np.dot(Kd, error_deriv).squeeze()

      msg = Twist()
      msg.linear.x = proportional[0] + derivative[0] + integral[0] 
      msg.angular.z = proportional[1] + derivative[1] + integral[1] 

      control_command = msg

      previous_error = error
      prev_time = curr_time
      pub.publish(control_command)

      print(error)

      if abs(error[0]) < 0.1 and abs(error[1]) < 0.1 : #TODO: what is our stopping condition/how do we know to go to the next waypoint?
        print("Moving to next waypoint in trajectory")
        return error

    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
      print("TF Error in Turtlebot Controller: " + e)
      pass
    # Use our rate object to sleep until it is time to publish again
    r.sleep()


def planning_callback(msg):
  try:
    trajectory = plan_curved_trajectory((msg.x, msg.y)) 

    for waypoint in trajectory:
      controller(waypoint)
  except rospy.ROSInterruptException as e:
    print("Exception thrown in planning callback: " + e)
    pass
      

# This is Python's sytax for a main() method, which is run by default
# when exectued in the shell
if __name__ == '__main__':
  # Check if the node has received a signal to shut down
  # If not, run the talker method

  #Run this program as a new node in the ROS computation graph 
  #called /turtlebot_controller.
  rospy.init_node('turtlebot_controller', anonymous=True)

  rospy.Subscriber('/goal_point', Point, planning_callback)
  
  rospy.spin()