#!/usr/bin/env python2
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

def joint_angle_values():
    with open ('/home/zafar/catkin_ws/src/Thesis/reading_angle_values/joint_angles_new1.txt') as fp:
        num_lines = len(fp.readlines())
        num_joints = 6
        arm_joints = np.ones((num_lines, num_joints))
        # print arm_joints
        fp.close()
    with open ('/home/zafar/catkin_ws/src/Thesis/reading_angle_values/joint_angles_new1.txt') as fp1:    
    
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
            
    # cam = cv2.VideoCapture(0)
    # retval, frame = cam.read()
    # if retval != True:
    #     raise ValueError("Can't read frame")
    # cv2.imwrite('/home/zafar/pictures/img'+str(i)+'.jpg', frame)
    # #cv2.imshow("img1", frame)
    # #cv2.waitKey()
    # cv2.destroyAllWindows()
    # cam.release()

    # rospy.sleep(0.1)
    group.stop()

