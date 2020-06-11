#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 17:25:42 2020
This is deepgaze code to get a pic and calculate the gaze
@author: zafar
"""

import tensorflow as tf
import cv2
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator
from PIL import Image
import time
 ####### Capturing image #########
'''cam = cv2.VideoCapture(1)
retval, frame = cam.read()
if retval != True:
    raise ValueError("Can't read frame")
flipHorizontal = cv2.flip(frame, 1)
cv2.imwrite('img2.jpg', flipHorizontal)
#cv2.imshow("img1", frame)
#cv2.waitKey()
cv2.destroyAllWindows()
cam.release() '''
######### Resizing image ##########
# while True:
counter = 0
while True:
    var = int (input())
    while var < 5:
        var = 10
        for i in range(1):
            #im1 = Image.open("img2.jpg")
            #time.sleep(0.01)
            cam = cv2.VideoCapture(1)
            retval, frame = cam.read()
            if retval != True:
                raise ValueError("Can't read frame")
            im1 = cv2.flip(frame,1)
            #im1 = Image.open("/home/zafar/zafar-rtech/Images_Houman/img"+str(i+1)+".jpg")
            img = cv2.resize(im1, (293, 293), interpolation = cv2.INTER_AREA)
            #im1 = im1.resize((293, 293), Image.BILINEAR)
            # image = im1
            image = cv2.imwrite("/home/zafar/Deepgaze_images_new2/image"+str(i+counter)+".jpg", im1)
            sess = tf.Session() #Launch the graph in a session.
            my_head_pose_estimator = CnnHeadPoseEstimator(sess) #Head pose estimation object
            my_head_pose_estimator.load_pitch_variables("/home/zafar/deepgaze/etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf")
            my_head_pose_estimator.load_yaw_variables("/home/zafar/deepgaze/etc/tensorflow/head_pose/yaw/cnn_cccdd_30k.tf")
            my_head_pose_estimator.load_roll_variables("/home/zafar/deepgaze/etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf")
            # image = cv2.imread("/home/zafar/Deepgaze_images/image"+str(i+counter+50)+".jpg") #Read the image with OpenCV
                    
            pitch = my_head_pose_estimator.return_pitch(img) #Evaluate the pitch angle using a CNN
            yaw = my_head_pose_estimator.return_yaw(img) #Evaluate the yaw angle using a CNN
            roll = my_head_pose_estimator.return_roll(img)
            print("Estimated pitch ..... " + str(pitch[0,0,0]))
            print("Estimated yaw ..... " + str(yaw[0,0,0]))
            print("Estimated roll ..... " + str(roll[0,0,0]))
                    
            with open('/home/zafar/catkin_ws/src/Thesis/reading_angle_values/New_Data/gaze_record2.txt', 'a+') as text_file:
                text_file.write(str(roll[0,0,0]) + '   '+str(pitch[0,0,0]) + '   '+str(yaw[0,0,0]))
                text_file.write('\n')
            retval = True
            cam.release()
        counter+=5

cv2.waitKey()
cv2.destroyAllWindows()
# cam.release()
