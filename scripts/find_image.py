#!/usr/bin/env python
# Python libs
import sys, time
# numpy and scipy
import numpy as np
from scipy.ndimage import filters
# OpenCV
import cv2
# Ros libraries
import roslib
import rospy
from cv_bridge import CvBridge
# Ros Messages
from sensor_msgs.msg import Image


VERBOSE=False

class image_feature:
    def __init__(self):
        '''Initialize ros publisher, ros subscriber'''
        # topic where we publish
        self.image_pub = rospy.Publisher("/find_image/img",
            Image, queue_size=10)
        # self.bridge = CvBridge()

        # subscribed Topic
        self.subscriber = rospy.Subscriber("/usb_cam/image_raw",
            Image, self.callback,  queue_size = 1)
        if VERBOSE :
            print "subscribed to /usb_cam/image_raw"


    def callback(self, ros_data):
        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(ros_data, "bgr8")
        cv2.imshow("listener", img)
        cv2.waitKey(3)
        print "find the image"

def main(args):
    '''Initializes and cleanup ros node'''
    ic = image_feature()
    rospy.init_node('image_feature', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS Image feature detector module"
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
