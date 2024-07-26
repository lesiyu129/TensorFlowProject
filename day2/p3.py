import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# boston_housing = keras.datasets.boston_housing
# (x_train, y_train), (x_test, y_test) = boston_housing.load_data(seed=42)
# print(x_train.shape, y_train)
# print(x_test, y_test)

df = pd.read_csv('housing.data', header=None)
ds = df.values

x = ds[:, :13]
y = ds[:, 13]
# print(x[:, 1])
# for i in range(13):
#     x[:, i] = (x[:, i] - x[:, i].min()) / (x[:, i].max() - x[:, i].min())

x = MinMaxScaler().fit_transform(x)
# 归一化


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

w = tf.Variable(tf.random.truncated_normal(
    [13, 1], mean=0, stddev=0.01, seed=42))
b = tf.Variable(tf.zeros([1, 1]))

lr = 0.001
loss_list = []
for i in range(15001):
    with tf.GradientTape() as tape:
        pred = tf.matmul(x_train, w) + b  # 预测值
        loss = tf.reduce_mean(tf.square(y_train - pred))  # 计算损失值

    loss_list.append(loss)
    grad = tape.gradient(loss, [w, b])
    w.assign_sub(lr * grad[0])
    b.assign_sub(lr * grad[1])

    if i % 100 == 0:
        print(f'第{i}次迭代,loss:{loss}')

# 预测
pred_test_y = tf.matmul(x_test, w) + b
pred_test_y = pred_test_y.numpy()


plt.figure("train_loss")
plt.title("train_loss")
plt.plot(loss_list, label='train_loss')
plt.legend()
plt.show()

plt.figure("predict")
plt.title("predict")
plt.xlabel("ground truth")
plt.ylabel("predict")
x = np.arange(1, 60)
y = x
plt.plot(x, y)
plt.scatter(y_test, pred_test_y, color='green', label='predict vs label')

plt.grid()
plt.legend()
plt.show()
