import tensorflow as tf
import keras
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
x, y = load_iris(return_X_y=True)
x = MinMaxScaler().fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=42))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=42))

lr = 0.01
train_loss_result = []
test_acc_result = []
loss_all = 0
for epoch in range(800):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1
            y = tf.nn.softmax(y)
            y_true = tf.one_hot(y_train, depth=3)

            cross_entropy = -tf.reduce_sum(y_true*tf.math.log(y), axis=1)
            loss = tf.reduce_mean(cross_entropy)
            loss_all += loss.numpy()

        grads = tape.gradient(loss, [w1, b1])
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])

        print("epoch:", epoch, "step:", step, "loss:", loss.numpy())

    train_loss_result.append(loss_all / 4)
    loss_all = 0

    # 测试集测试
    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)
        pred = tf.cast(pred, dtype=y_test.dtype)
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_number += x_test.shape[0]

    acc = total_correct / total_number
    test_acc_result.append(acc)
    print("testAcc", acc)

plt.title("Loss Function", fontsize=24)
plt.xlabel("epochs", fontsize=14)
plt.ylabel("loss", fontsize=14)
plt.plot(train_loss_result, label="Train Loss")
plt.legend()
plt.show()


plt.title("Accuracy", fontsize=24)
plt.xlabel("epochs", fontsize=14)
plt.ylabel("acc", fontsize=14)
plt.plot(test_acc_result, label="Test Acc")
plt.legend()
plt.show()
