cd read#!/usr/bin/env python2
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import JointState
from actionlib_msgs.msg import GoalStatusArray
import os
import tensorflow as tf
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator
from PIL import Image
import time

def joint_angle_values():
    with open ('/home/zafar/catkin_ws/src/Thesis/reading_angle_values/joint_angles.txt') as fp:
        num_lines = len(fp.readlines())
        num_joints = 6
        arm_joints = np.ones((num_lines, num_joints))
        # print arm_joints
        fp.close()
    with open ('/home/zafar/catkin_ws/src/Thesis/reading_angle_values/joint_angles.txt') as fp1:    
    
        for i in range(num_lines):
            line = fp1.readline()
            x = np.fromstring(line.strip(), dtype=float, sep=' ')
            for j in range(num_joints):
                arm_joints[i,j] = x[j]
        return arm_joints        

arm_joints = joint_angle_values()
moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('move_group_python_interface_tutorial',
                anonymous=True)

robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()
group_name = "manipulator"
group = moveit_commander.MoveGroupCommander(group_name)
for i in range (arm_joints.shape[0]):
    joint_goal = group.get_current_joint_values()
    joint_goal[0] = arm_joints[i,0]
    joint_goal[1] = arm_joints[i,1]
    joint_goal[2] = arm_joints[i,2]
    joint_goal[3] = arm_joints[i,3]
    joint_goal[4] = arm_joints[i,4]
    joint_goal[5] = arm_joints[i,5]
    group.go(joint_goal, wait=True)
    with open ('/home/zafar/catkin_ws/src/Thesis/reading_angle_values/end_effector_coordinates.txt', 'a+') as text_file:
        pose = group.get_current_pose().pose
        print pose
        # print ("After Pose")
        # print str(pose)
        text_file.write(str(pose))
        text_file.write('\n')

    #im1 = Image.open("img2.jpg")
    #time.sleep(0.01)
    cam = cv2.VideoCapture(0)
    retval, frame = cam.read()
    if retval != True:
        raise ValueError("Can't read frame")
    im1 = cv2.flip(frame,1)
    #im1 = Image.open("/home/zafar/zafar-rtech/Images_Houman/img"+str(i+1)+".jpg")
    im1 = cv2.resize(im1, (293, 293), interpolation = cv2.INTER_AREA)
    #im1 = im1.resize((293, 293), Image.BILINEAR)
    image = im1
    # image = cv2.imwrite("/home/zafar/.config/spyder/image.jpg", im1)
    sess = tf.Session() #Launch the graph in a session.
    my_head_pose_estimator = CnnHeadPoseEstimator(sess) #Head pose estimation object
    my_head_pose_estimator.load_pitch_variables("/home/zafar/deepgaze/etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf")
    my_head_pose_estimator.load_yaw_variables("/home/zafar/deepgaze/etc/tensorflow/head_pose/yaw/cnn_cccdd_30k.tf")
    my_head_pose_estimator.load_roll_variables("/home/zafar/deepgaze/etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf")
    # image = cv2.imread("/home/zafar/.config/spyder/image.jpg") #Read the image with OpenCV
            
    pitch = my_head_pose_estimator.return_pitch(image) #Evaluate the pitch angle using a CNN
    yaw = my_head_pose_estimator.return_yaw(image) #Evaluate the yaw angle using a CNN
    roll = my_head_pose_estimator.return_roll(image)
    # print("Estimated pitch ..... " + str(pitch[0,0,0]))
    # print("Estimated yaw ..... " + str(yaw[0,0,0]))
    # print("Estimated roll ..... " + str(roll[0,0,0]))
            
    with open('/home/zafar/catkin_ws/src/Thesis/reading_angle_values/gaze_record.txt', 'a+') as text_file:
        text_file.write(str(roll[0,0,0]) + '   '+str(pitch[0,0,0]) + '   '+str(yaw[0,0,0]))
        text_file.write('\n')
    retval = True
    cam.release()

    rospy.sleep(0.1)
    group.stop()

