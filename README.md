Run ```catkin_make``` to initialize workspace.

To use workspace, first run 
```source ~ee106a/sawyer_setup.bash``` and ```source devel/setup.bash```. Should run in every new terminal.

To start RealSense, run 
```
roslaunch realsense2_camera rs_camera.launch mode:=Manual color_width:=424 \
color_height:=240 depth_width:=424 depth_height:=240 align_depth:=true \
depth_fps:=30 color_fps:=30
```

Start Sawyer head camera:
```
rosrun intera_examples camera_display.py -c head_camera
```

Start AR tag tracking:
```
roslaunch sawyer_full_stack sawyer_camera_track.launch
```

Start intera action server for Sawyer
```
rosrun intera_interface joint_trajectory_action_server.py
```

Static transform from Realsense Camera frame to Sawyer frame
```
rosrun tf static_transform_publisher 0 0 0 0 0 0 base camera_link 100
```

Start sawyer MoveIt and Rviz window
```
roslaunch sawyer_moveit_config sawyer_moveit.launch electric_gripper:=true
```
(Rviz setup: add Image with "detected_cup" and PointStamped with "goal_point_stamped")

Run object detector:
```
cd src/sawyer_full_stack/src/perception/src
python traj_test2.py
```

Run trajectory: to move Sawyer end effector towards ball position
```
python main.py -task "line"
```

Run trajectory thru MPC:
```
cd src/sawyer_full_stack/src/controllers
python mpc_controller.py
```

Resetting Sawyer Position (tuck):
```
roslaunch sawyer_full_stack custom_sawyer_tuck.launch
```

