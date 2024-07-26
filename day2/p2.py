# 自动微分求导
import tensorflow as tf

x = tf.Variable(4.0)
y = tf.Variable(2.0)

with tf.GradientTape(persistent=True) as tape:
    z = x+y
    w = tf.pow(x, 3)

dz_dy = tape.gradient(z, y)
dz_dx = tape.gradient(z, x)
dw_dx = tape.gradient(w, x)

tf.print(dz_dy)
tf.print(dz_dx)
tf.print(dw_dx)
del tape
