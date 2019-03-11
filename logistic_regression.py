import tensorflow as tf
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# generate data
'''
    return a samples from "standard normal" distribution.
    parameter is return shape
'''
x_data = np.random.randn(20000, 3)
w_real = [0.3, 5.0, 0.1]
b_real = -0.2

hypothesis = sigmoid(np.matmul(w_real, x_data.T) + b_real)  # to use logistic regression
y_data = np.random.binomial(1, hypothesis)

NUM_STEPS = 50

g = tf.Graph()

with g.as_default():
    x = tf.placeholder(tf.float32, shape=[None, 3])
    y_true = tf.placeholder(tf.float32, shape=None)

    with tf.name_scope('inference') as scope:
        w = tf.Variable([[0, 0, 0]], dtype=tf.float32, name='weights')
        b = tf.Variable(0, dtype=tf.float32, name='bias')
        y_pred = tf.matmul(w, tf.transpose(x)) + b

    with tf.name_scope('loss') as scope:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        loss = tf.reduce_mean(loss)

    with tf.name_scope('train') as scope:
        learning_rate = 0.5
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(NUM_STEPS):
            sess.run(train, {x: x_data, y_true: y_data})

        print(50, sess.run([w, b]))
