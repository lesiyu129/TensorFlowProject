# 张量的索引与切片

import tensorflow as tf

t1 = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 索引会降维
print(t1[0].numpy())

# 切片
print("所有元素", t1[:].numpy())
print("前3个元素", t1[:3].numpy())
print("后3个元素", t1[-3:].numpy())
print("中间3个元素", t1[2:7].numpy())
print("步长2", t1[::2].numpy())
print("逆序", t1[::-1].numpy())

# 三维张量的切片
t3 = tf.constant(
    [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]]
)
print(t3)
print(t3[:, :, 2])  # 第一个维度、第二个维度（行）、第二个元素（列） 维度会降级
