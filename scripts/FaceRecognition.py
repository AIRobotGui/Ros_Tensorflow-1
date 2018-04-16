#!/usr/bin/env python
"""
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

    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/find_face/img", Image, callback)
    
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
if __name__ == '__main__':
    init()

"""

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorflow as tf




if __name__ == '__main__':
    try:
        RosTensorFlow()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("RosTensorFlow has started.")
