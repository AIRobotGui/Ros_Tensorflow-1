#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorflow as tf

def convolutional_layer(data, kernel_size, bias_size, pooling_size):
    kernel = tf.get_variable("conv", kernel_size, initializer=tf.random_normal_initializer())
    bias = tf.get_variable('bias', bias_size, initializer=tf.random_normal_initializer())

    conv = tf.nn.conv2d(data, kernel, strides=[1, 1, 1, 1], padding='SAME')
    linear_output = tf.nn.relu(tf.add(conv, bias))
    pooling = tf.nn.max_pool(linear_output, ksize=pooling_size, strides=pooling_size, padding="SAME")
    return pooling


def linear_layer(data, weights_size, biases_size):
    weights = tf.get_variable("weigths", weights_size, initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases", biases_size, initializer=tf.random_normal_initializer())
    return tf.add(tf.matmul(data, weights), biases)



def convolutional_neural_network(data):


    data = tf.reshape(data, [-1, 57, 47, 1])

    # 经过第一层卷积神经网络后，得到的张量shape为：[batch, 29, 24, 32]
    with tf.variable_scope("conv_layer1") as layer1:
        layer1_output = convolutional_layer(
            data=data,
            kernel_size=kernel_shape1,
            bias_size=bias_shape1,
            pooling_size=[1, 2, 2, 1]
        )
    # 经过第二层卷积神经网络后，得到的张量shape为：[batch, 15, 12, 64]
    with tf.variable_scope("conv_layer2") as layer2:
        layer2_output = convolutional_layer(
            data=layer1_output,
            kernel_size=kernel_shape2,
            bias_size=bias_shape2,
            pooling_size=[1, 2, 2, 1]
        )
    with tf.variable_scope("full_connection") as full_layer3:
        # 讲卷积层张量数据拉成2-D张量只有有一列的列向量
        layer2_output_flatten = tf.contrib.layers.flatten(layer2_output)
        print("vec:=====:",layer2_output_flatten)
        layer3_output = tf.nn.relu(
            linear_layer(
                data=layer2_output_flatten,
                weights_size=full_conn_w_shape,
                biases_size=full_conn_b_shape
            )
        )

    with tf.variable_scope("output") as output_layer4:
        output = linear_layer(
            data=layer3_output,
            weights_size=out_w_shape,
            biases_size=out_b_shape
        )

    vec = output
    return output;


class RosTensorFlow():
    def __init__(self):

        rospy.init_node('rostensorflow')

    # Set the shutdown function (stop the robot)
        rospy.on_shutdown(self.shutdown)

        model_path = rospy.get_param("~model_path", "")
        image_topic = rospy.get_param("~image_topic", "")

        self._cv_bridge = CvBridge()

        self.x = tf.placeholder(tf.float32, [None,64,64,1], name="x")
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
        cv_image_64 = cv2.resize(cv_image_binary,(64,64))
        np_image = np.reshape(cv_image_64,(1,64,64,1))
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

