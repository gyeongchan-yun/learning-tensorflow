import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import os

DATA_DIR = os.path.join(os.getcwd(), 'data')
NUM_STEPS = 1000
MINIBATCH_SIZE = 100

data = input_data.read_data_sets(DATA_DIR, one_hot=True)

# match one vector with 28 x 28 pixels
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))

y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)  # matrix multiplication

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=y_pred, labels=y_true))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for _ in range(NUM_STEPS):
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        sess.run(train_step, feed_dict={x: batch_xs, y_true: batch_ys})

    accuracy = sess.run(accuracy, feed_dict={
                        x: data.test.images, y_true: data.test.labels})

print("Accuracy: {:.4}%".format(accuracy * 100))
