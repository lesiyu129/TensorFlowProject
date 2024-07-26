# 变更张量的维度
import tensorflow as tf

t2 = tf.Variable(tf.constant([[1], [2], [3]]))
print(t2.shape)
t1 = tf.reshape(t2, [1, 3])
print(t1)
print(t1.shape)

# 向量乘法
x = tf.Variable(tf.constant([[1, 2, 3]]))
print(x)
x = tf.reshape(x, [3, 1])
print(x)
y = tf.range(1, 5)

print(tf.multiply(x, y))


# 矩阵相乘
x = tf.constant([[1, 2, 3], [4, 5, 6]])
print(x.shape)
y = tf.constant([[1, 2], [3, 4], [5, 6]])
print(y.shape)
print(tf.matmul(x, y))

# 向量计算 axis=0 列 axis=1 行
x = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
tf.print(tf.reduce_sum(x, axis=0))
tf.print(tf.reduce_sum(x, axis=1))
tf.print(tf.reduce_max(x, axis=0))
tf.print(tf.reduce_max(x, axis=1))
tf.print(tf.reduce_min(x, axis=0))
tf.print(tf.reduce_min(x, axis=1))

tf.print("argmax", tf.argmax(x))
tf.print("argmin", tf.argmin(x))
