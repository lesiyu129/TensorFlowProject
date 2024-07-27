import tensorflow as tf
from keras import Sequential, layers, activations
from keras import datasets

mnist = datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0  # 归一化（0~1）
model = Sequential(
    [
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation=activations.relu),
        layers.Dense(64, activation=activations.relu),
        layers.Dropout(0.2),
        layers.Dense(10, activation=activations.softmax)
    ]
)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=7)
model.evaluate(x_test, y_test, verbose=2)

result = model.predict(x_test[:5])
print(y_test[:5])
for r in result:
    print(tf.argmax(r).numpy())

model.summary()
