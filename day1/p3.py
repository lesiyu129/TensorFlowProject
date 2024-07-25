# 变更张量的维度
import tensorflow as tf

t2 = tf.Variable(tf.constant([[1], [2], [3]]))
print(t2.shape)
t1 = tf.reshape(t2, [1, 3])
print(t1)
print(t1.shape)
