import tensorflow as tf

with tf.Session() as sess:
    c = tf.linspace(0.0, 4.0, 5)
    print("'c': {}".format(c.eval()))
