# importing numpy is required due to bug  in tensorflow
import numpy
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print sess.run(hello)
