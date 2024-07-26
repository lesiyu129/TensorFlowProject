import tensorflow as tf

a = tf.Variable([2.0, 3.0])
print(a)
a.assign([1.0, 2.0])
print(a)
a.assign_add([1.0, 2.0])
print(a)
