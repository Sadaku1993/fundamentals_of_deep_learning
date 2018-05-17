#coding:utf-8

"""
Sessionの使い方を確認
"""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

x = tf.placeholder(tf.float32, name="x", shape=[None, 5])
W = tf.Variable(tf.random_uniform([5, 10], -1, 1), name="W")
b = tf.Variable(tf.zeros([10]), name="biases")
output = tf.matmul(x, W) + b

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

x_vals = np.array([[1., 2., 3., 4., 5.],
                   [1., 2., 3., 4., 5.]])

feed_dict = { x : x_vals }

print(sess.run(output, feed_dict=feed_dict))


