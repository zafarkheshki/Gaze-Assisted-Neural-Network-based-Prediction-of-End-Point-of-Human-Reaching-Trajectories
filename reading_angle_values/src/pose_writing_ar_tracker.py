#!/usr/bin/env python2

import numpy as np
import rospy
from ar_track_alvar_msgs.msg import AlvarMarkers
from geometry_msgs.msg import Pose
from actionlib_msgs.msg import GoalStatusArray
import os


record = False
writing = False

def AR_state_callback(msg):
    global record
    global writing
    # print ("Here I am")
    # print(msg.markers[0].pose.pose)
    
    with open ('/home/zafar/catkin_ws/src/Thesis/reading_angle_values/ar_tracker_pose.txt', 'a+') as text_file:
        print('Writing to file')
        for markers in msg.markers:
            text_file.write(str(msg.markers[0].pose.pose) + ' ')
            text_file.write('\n')
            print('Im writing')

    # if record:
    #     with open ('/home/zafar/catkin_ws/src/Thesis/reading_angle_values/ar_tracker_pose.txt', 'a+') as text_file:
    #         print('Writing to file')
            # if not writing:
            #     writing = True
            #     for name in msg.position:
            #         text_file.write(str(position) + ' ')
            #     text_file.write('\n\n')
            #     print('I am writing')
            # for angle in msg.position:
            #     text_file.write(str(angle) + ' ')
            # text_file.write('\n\n')

        # write to file this is your task

# def goal_status_callback(msg):
#     global record
#     global writing

#     # print(msg.status_list)
#     if len(msg.status_list) > 0:
#         status = msg.status_list[len(msg.status_list)-1].status

#         print status

#         if status == 1:
#             record = True
#         else:
#             record = False
#             writing = False
#     # print("=====================")

if __name__ == "__main__":
    rospy.init_node('pose_writing_ar_tracker', anonymous=True)
    print("hello I am in writer node")
    rospy.Subscriber('/ar_pose_marker', AlvarMarkers, AR_state_callback)
    # rospy.Subscriber('/move_group/status', GoalStatusArray, goal_status_callback)
    rospy.spin()
