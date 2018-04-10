#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

def callback(imgmsg):
    print("The model has been generated!")

def listener():

    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/find_face/img", Image, callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
if __name__ == '__main__':
    listener()
