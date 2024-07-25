import tensorflow as tf

t0 = tf.constant(4)  # 0维张量
t1 = tf.constant([1.0, 2.0, 3.0])  # 1维张量
t2 = tf.constant(
    [
        [
            1.0,
            2.0,
        ],
        [3.0, 4.0],
        [5.0, 6.0],
    ]
)  # 2维张量
t3 = tf.constant(
    [
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
    ]
)  # 3维张量
print(t0)
print(t1)
print(t2)
print(t3)

# 张量转换成数组
arr0 = t0.numpy()
arr1 = t1.numpy()
arr2 = t2.numpy()
arr3 = t3.numpy()
print(arr0)
print(arr1)
print(arr2)
print(arr3)

# 数组转换成张量
t0 = tf.convert_to_tensor(arr0)
t1 = tf.convert_to_tensor(arr1)
t2 = tf.convert_to_tensor(arr2)
t3 = tf.convert_to_tensor(arr3)
print(t0)
print(t1)
print(t2)
print(t3)
