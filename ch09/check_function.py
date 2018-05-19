import tensorflow as tf
import numpy as np

sess = tf.Session()

a = [[0,1],[1,0],[0,1],[0,1],[1,0]]
a = np.array(a)
print(a)
print(a.shape)

b = [0, 1, 0, 0, 0]
b = np.array(b)
print(b)
print(b.shape)

x = tf.placeholder(tf.int32, shape=[None, 2])
y = tf.placeholder(tf.int32, shape=[None])

output_index = (tf.range(0, tf.shape(x)[0] * tf.shape(x)[1])) + y

print(sess.run(output_index, feed_dict={x : a, y : b}))
