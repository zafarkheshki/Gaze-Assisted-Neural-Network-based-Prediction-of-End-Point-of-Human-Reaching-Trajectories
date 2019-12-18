#!/usr/bin/env python2

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from actionlib_msgs.msg import GoalStatusArray
import os


record = False
writing = False

# global line
# line = []
global arm_joints
#arm_joints = np.ones((num_lines, num_joints))
with open ('/home/zafar/zafar-rtech/src/reading_angle_values/joint_angles.txt') as fp:
    rows = len(fp.readlines())
    columns = 6
    #np.mart
    print (rows)
    # print count.toString()
    # for i in fp.read().split(' '):
    #     line.append(i)

    






# global line
# line = []
# with open ('/home/zafar/zafar-rtech/src/reading_angle_values/joint_angles.txt') as fp:
#     for i in fp.read().split(' '):
#         line.append(i)
# del line[-1]
# line = map (float, line)

# print line
#    while line:
#        if(line.strip() == ''):
#         line = fp.readline()

       
#        x = np.fromstring(line.strip(), dtype=float, sep=' ')
#        print x

# def joint_state_callback(msg):
#     global record
#     global writing
#     if record:
#         with open ('/home/zafar/zafar-rtech/src/reading_angle_values/joint_angles.txt', 'a+') as text_file:
#             print('Writing to file')
#             if not writing:
#                 writing = True
#                 for name in msg.name:
#                     text_file.write(name + ' ')
#                 text_file.write('\n\n')
            
#             for angle in msg.position:
#                 text_file.write(str(angle) + ' ')
#             text_file.write('\n\n')

#         # write to file this is your task

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

# if __name__ == "__main__":
#     rospy.init_node('file_writer', anonymous=True)
#     print("hello I am in writer node")
#     rospy.Subscriber('/joint_states', JointState, joint_state_callback)
#     rospy.Subscriber('/move_group/status', GoalStatusArray, goal_status_callback)
#     rospy.spin()
# moveit_commander.roscpp_initialize(sys.argv)
# rospy.init_node('move_group_python_interface_tutorial', anonymous=True)
# robot = moveit_commander.RobotCommander()
# scene = moveit_commander.PlanningSceneInterface()
# group_name = "manipulator"
# group = moveit_commander.MoveGroupCommander(group_name)
# i = 0
# # line  = np.asarray(line)
# print (line)
#     # while (i<np.size(line)): 
#     #     joint_goal = group.get_current_joint_values()
#     #     joint_goal[0] = line[i]
#     #     joint_goal[1] = line[i+1]
#     #     joint_goal[2] = line[i+2]
#     #     joint_goal[3] = line [i+3]
#     #     joint_goal[4] = line [i+4]
#     #     joint_goal[5] = line [i+5]
#     #     i = i+6
#     #     rospy.sleep(2)
# # The go command can be called with joint values, poses, or without any
# # parameters if you have already set the pose or joint target for the group
# group.go(joint_goal, wait=True)

# # Calling ``stop()`` ensures that there is no residual movement
# group.stop()