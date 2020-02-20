#!/usr/bin/env python2

import numpy as np
import cv2
import rospy
from ar_track_alvar_msgs.msg import AlvarMarkers
from geometry_msgs.msg import Pose
from actionlib_msgs.msg import GoalStatusArray
import os


class ARStateHandler(object):

    def __init__(self):
        print("hello I am in writer node")
        rospy.Subscriber('/ar_pose_marker', AlvarMarkers,
                         self.AR_state_callback)
        self.file = open(
            '/home/zafar/catkin_ws/src/Thesis/reading_angle_values/Data_collection/ar_tracker_pose.txt', 'a+')
        self.is_written = False

    def AR_state_callback(self, msg):
    
        if len(msg.markers) > 0:
            print(msg.markers[0].pose.pose.position.x)
            print(str(msg.markers[0].pose.pose))
            self.file.write(str(msg.markers[0].pose.pose) + ' ')
            self.file.write('\n')
            self.is_written = True
            # print('Im writing')

    def on_shutdown(self):
        if self.is_written:
            eof = 'x '*7
            print("Ended file with"+eof)
            self.file.write(eof+'\n')
        self.file.close()

    def run(self):
        rospy.on_shutdown(self.on_shutdown)
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node('pose_writing_ar_tracker', anonymous=True)
    obj = ARStateHandler()
    obj.run()
