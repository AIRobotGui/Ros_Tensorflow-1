#!/usr/bin/env python
import rospy
import tensorflow as tf
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import sys
import os
import numpy as np
import random
from cv_bridge import CvBridge
from sklearn.model_selection import train_test_split


def callback(imgmsg):
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(imgmsg, "bgr8")
    print('ID: 1510243221 '+'Name : Xu Yiming' + ' is coming!')
    cv2.waitKey(3)

def init():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/find_face/img", Image, callback)
    
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
if __name__ == '__main__':
    init()
