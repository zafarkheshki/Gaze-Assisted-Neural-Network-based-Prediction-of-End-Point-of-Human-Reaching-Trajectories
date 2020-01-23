#!/usr/bin/env python2

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from actionlib_msgs.msg import GoalStatusArray
import os


record = False
writing = False

def joint_state_callback(msg):
    global record
    global writing
    if record:
        with open ('/home/zafar/catkin_ws/src/Thesis/reading_angle_values/joint_angles.txt', 'a+') as text_file:
            print('Writing to file')
            if not writing:
                writing = True
                for name in msg.name:
                    text_file.write(name + ' ')
                text_file.write('\n\n')
            
            for angle in msg.position:
                text_file.write(str(angle) + ' ')
            text_file.write('\n\n')

        # write to file this is your task

def goal_status_callback(msg):
    global record
    global writing

    # print(msg.status_list)
    if len(msg.status_list) > 0:
        status = msg.status_list[len(msg.status_list)-1].status

        print status

        if status == 1:
            record = True
        else:
            record = False
            writing = False
    # print("=====================")

if __name__ == "__main__":
    rospy.init_node('file_writer', anonymous=True)
    print("hello I am in writer node")
    rospy.Subscriber('/joint_states', JointState, joint_state_callback)
    rospy.Subscriber('/move_group/status', GoalStatusArray, goal_status_callback)
    rospy.spin()
