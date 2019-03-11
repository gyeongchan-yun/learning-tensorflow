import tensorflow as tf

hello = tf.constant("Hello")
world = tf.constant(" World!")

sentence = hello + world

with tf.Session() as sess:
    msg = sess.run(sentence)

print(sentence)
