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


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], 
                      padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def makeCNN(x,keep_prob):
    # --- define CNN model
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return y_conv



class RosTensorFlow():
    def __init__(self):

    rospy.init_node('rostensorflow')

    # Set the shutdown function (stop the robot)
        rospy.on_shutdown(self.shutdown)

        model_path = rospy.get_param("~model_path", "")
        image_topic = rospy.get_param("~image_topic", "")

        self._cv_bridge = CvBridge()

        self.x = tf.placeholder(tf.float32, [None,28,28,1], name="x")
        self.keep_prob = tf.placeholder("float")
        self.y_conv = makeCNN(self.x,self.keep_prob)

        self._saver = tf.train.Saver()
        self._session = tf.InteractiveSession()

        init_op = tf.global_variables_initializer()
        self._session.run(init_op)

        self._saver.restore(self._session, model_path+"/model.ckpt")

        self._sub = rospy.Subscriber(image_topic, Image, self.callback, queue_size=1)
        #self._pub = rospy.Publisher('result', Int16, queue_size=1)
        self._pub = rospy.Publisher('xfwords', String, queue_size=1)


    def callback(self, image_msg):
        cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
        cv_image_gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        ret,cv_image_binary = cv2.threshold(cv_image_gray,128,255,cv2.THRESH_BINARY_INV)
        cv_image_28 = cv2.resize(cv_image_binary,(28,28))
        np_image = np.reshape(cv_image_28,(1,28,28,1))
        predict_num = self._session.run(self.y_conv, feed_dict={self.x:np_image,self.keep_prob:1.0})
        answer = np.argmax(predict_num,1)
        rospy.loginfo('%d' % answer)
        self._pub.publish(str(answer))
        rospy.sleep(3) 

    def shutdown(self):
        rospy.loginfo("Stopping the tensorflow nnist...")
        rospy.sleep(1)   

if __name__ == '__main__':
    try:
        RosTensorFlow()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("RosTensorFlow has started.")
