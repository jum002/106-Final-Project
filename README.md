# 106 Final Project

## Setup

Run ```catkin_make``` to initialize workspace.
To use workspace, first run ```source devel/setup.bash```.
To start RealSense, run 
```
roslaunch realsense2_camera rs_camera.launch mode:=Manual color_width:=424 \
color_height:=240 depth_width:=424 depth_height:=240 align_depth:=true \
depth_fps:=6 color_fps:=6
```

## ROS packages

- perception: use RealSense camera for stereo vision
    - goal point: x depth, y horizontal, z vertical
- plannedcntrl: reference PID controller setup using perception to control turtlebot