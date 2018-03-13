#!/usr/bin/env python
# Python libs
import sys, time
# numpy and scipy
import numpy as np
from scipy.ndimage import filters
# OpenCV
import cv2
import random
# Ros libraries
import roslib
import rospy
import dlib
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

    def relight(img, light=1, bias=0):
        w = img.shape[1]
        h = img.shape[0]
        #image = []
        for i in range(0,w):
            for j in range(0,h):
                for c in range(3):
                    tmp = int(img[j,i,c]*light + bias)
                    if tmp > 255:
                        tmp = 255
                    elif tmp < 0:
                        tmp = 0
                    img[j, i, c] = tmp
        return img


    def callback(self, ros_data):
        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(ros_data, "bgr8")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dets = detector(gray_img, 1)


        for i, d in enumerate(dets):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0

            face = img[x1:y1, x2:y2]
            face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
            face = cv2.resize(face, (size,size))
            #cv2.imshow('image', face)
            #cv2.imwrite(output_dir+'/'+str(index)+'.jpg', face)
            pub.publish(face)
            cv.imshow("face",face)

        #bridge = CvBridge()
        #img = bridge.imgmsg_to_cv2(ros_data, "bgr8")
        #cv2.imshow("listener", img)
        #cv2.waitKey(3)
        #print "find the image"

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
