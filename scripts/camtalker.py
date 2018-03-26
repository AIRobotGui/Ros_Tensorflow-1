#!/usr/bin/env python
# license removed for brevity
import rospy
import datetime
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge


def talker():
    pub = rospy.Publisher('/icamera/image', Image, queue_size=1) 
    pub2 = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('talker', anonymous=True) 
    rate = rospy.Rate(30)
    bridge = CvBridge()
    Video = cv2.VideoCapture(0)
    while not rospy.is_shutdown():
        ret, img = Video.read()
        #cv2.imshow("talker", img)
        cv2.waitKey(3)
        pub.publish(bridge.cv2_to_imgmsg(img, "bgr8"))
        hello_str = "find the camera %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub2.publish(hello_str)
        #nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        #print(""+nowTime)
        rate.sleep()
   

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
