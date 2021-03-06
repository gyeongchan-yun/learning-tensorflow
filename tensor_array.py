import tensorflow as tf
import numpy as np

c = tf.constant([[1, 2, 3],
                 [4, 5, 6]])

print("c.get_shape(): {}".format(c.get_shape()))

c = tf.constant(np.array([
    [[1, 2, 3],
     [4, 5, 6]],

    [[1, 1, 1],
     [2, 2, 2]]
]))

print("3D numpy array shape: {}".format(c.get_shape()))
