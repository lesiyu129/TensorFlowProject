from tensorflow.python.keras import layers, Sequential, activations
import tensorflow as tf

model = Sequential([
    layers.Dense(3, activation=None),
    layers.ReLU(),
    layers.Dense(2, activation=None),
    layers.ReLU()
])

x = tf.random.normal([4, 3])
tf.print(x)

y = model(x)
tf.print(y)

model.summary()
