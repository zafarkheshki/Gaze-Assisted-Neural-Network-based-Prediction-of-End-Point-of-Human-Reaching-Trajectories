#!/usr/bin/env python2

import numpy as np
import cv2
import rospy
from ar_track_alvar_msgs.msg import AlvarMarkers
from geometry_msgs.msg import Pose
from actionlib_msgs.msg import GoalStatusArray
import os


def AR_state_callback(msg):
   
    with open ('/home/zafar/catkin_ws/src/Thesis/reading_angle_values/Data_collection/ar_tracker_pose1.txt', 'a+') as text_file:
        # print('Writing to file')
        for marker in msg.markers:
            # cam = cv2.VideoCapture(0)
            # retval, frame = cam.read()
            # # if retval != True:
            # #     raise ValueError("Can't read frame")
            # im1 = cv2.flip(frame,1)
            # image = cv2.imwrite("/home/zafar/AR_tracker_images/image"+str(marker)+".jpg", im1)
            # print(len(msg.markers))
            print(msg.markers[0].pose.pose.position.x)
            print(str(msg.markers[0].pose.pose))
            text_file.write(str(msg.markers[0].pose.pose) + ' ')
            text_file.write('\n')
            # print('Im writing')

if __name__ == "__main__":
    rospy.init_node('pose_writing_ar_tracker', anonymous=True)
    print("hello I am in writer node")
    rospy.Subscriber('/ar_pose_marker', AlvarMarkers, AR_state_callback)
    rospy.spin()
