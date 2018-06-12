#coding:utf-8

"""
ロジステック回帰モデルの実装

"""

import tensorflow as tf
import time, shutil, os
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 100
batch_size = 100
display_step = 1

# 与えられたミニバッチに対して、出力クラスの確率分布を生成
def inference(x):
    # weight_init = tf.random_normal_initializer(stddev=(2.0 / 784)**0.5)
    # bias_init = tf.constant_initializer(value=0)
    init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", [784, 10], initializer=init)
    b = tf.get_variable("b", [10], initializer=init)
    output = tf.nn.softmax(tf.matmul(x, W) + b)
    tf.summary.histogram("weights", W)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("output", output)
    return output
   
# 損失関数(交差エントロピー誤差)を算出
def loss(output, y):
    # tf.reduce_sum : 数値の総和を求める関数 axis=1でj要素の総和となる
    # tf.reduce_mean : 与えたリストに入っている数値の平均値を求める関数 
    dot_product = y * tf.log(output)
    xentropy = -tf.reduce_sum(dot_product, axis=1)
    loss = tf.reduce_mean(xentropy)
    return loss

# モデルの勾配を求め、モデルを更新する
def training(cost, global_step):
    tf.summary.scalar("cost", cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op

# モデルの性能を評価する
def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("validation error", (1.0 - accuracy))
    return accuracy



if __name__ == "__main__":
    # Remove old summaries and checkpoints
    if os.path.exists("logistic_logs"):
        shutil.rmtree("logistic_logs")

    with tf.Graph().as_default():
        x = tf.placeholder("float", [None, 784])
        y = tf.placeholder("float", [None, 10])

        output = inference(x)
        cost = loss(output, y)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = training(cost, global_step)
        eval_op = evaluate(output, y)
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(
            "logistic_logs",
            graph_def=sess.graph_def
        )
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                minibatch_x, minibatch_y = mnist.train.next_batch(batch_size)
                # Fit training using batch data
                sess.run(train_op, feed_dict={x: minibatch_x, y: minibatch_y})
                # Compute average loss
                avg_cost += sess.run(
                    cost, feed_dict={x: minibatch_x, y: minibatch_y}
                ) / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch: {:04d} cost: {:.9f}".format(epoch+1, avg_cost))
                accuracy = sess.run(eval_op, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
                print("Validation Error: {}".format(1 - accuracy))
                summary_str = sess.run(summary_op, feed_dict={x: minibatch_x, y: minibatch_y})
                summary_writer.add_summary(summary_str, sess.run(global_step))
                saver.save(sess, os.path.join("logistic_logs", "model-checkpoint"), global_step=global_step)

        print("Optimization Finished!")
        accuracy = sess.run(eval_op, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Test Accuracy: {}".format(accuracy))
