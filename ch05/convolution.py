#coding:utf-8

import tensorflow as tf
import numpy as np

sess = tf.Session()

x_shape = [1, 4, 4, 1]
x_val = np.random.uniform(size=x_shape)

x_data = tf.placeholder(tf.float32, shape=x_shape)

my_filter = tf.constant(0.25, shape=[2, 2, 1, 1])
my_strides = [1, 2, 2, 1]
mov_avg_layer = tf.nn.conv2d(x_data, my_filter, my_strides,
                             padding='SAME', name='Moving_Avg_Window')

print(x_val)
output = sess.run(mov_avg_layer, feed_dict={x_data:x_val})
print(output)
print(output.shape)
