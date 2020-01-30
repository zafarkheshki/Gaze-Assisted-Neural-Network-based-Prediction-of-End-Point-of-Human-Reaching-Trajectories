#!/usr/bin/env python2

import numpy as np
import rospy
from ar_track_alvar_msgs.msg import AlvarMarkers
from geometry_msgs.msg import Pose
from actionlib_msgs.msg import GoalStatusArray
import os

global pos
def AR_state_callback(msg):
   
    with open ('/home/zafar/catkin_ws/src/Thesis/reading_angle_values/ar_tracker_pose.csv', 'w+') as csv_file:
    # f = open('/home/usman/usman-ros/src/skywalker/skywalker/src/joint_states.csv', 'w')
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # print('Writing to file')
        for marker in msg.markers:
            # print(len(msg.markers))
            # print(msg.markers[0].pose.pose.position.x)
            pos = msg.markers[0].pose.pose
            print(str(msg.markers[0].pose.pose))
            writer.writerow([pos.position.x, pos.position.y, pos.position.z, pos.orientation.x, pos.orientation.y, pos.orientation.z, pos.orientation.w])
            # csv_file.write('\n')
            # print('Im writing')

if __name__ == "__main__":
    rospy.init_node('pose_writing_ar_tracker', anonymous=True)
    print("hello I am in writer node")
    rospy.Subscriber('/ar_pose_marker', AlvarMarkers, AR_state_callback)
    rospy.spin()
