import pathlib
import numpy as np
import tensorflow as tf
from keras import Sequential, layers, optimizers, losses, utils
import numpy as np
import matplotlib.pyplot as plt

import PIL

print(tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("TensorFlow detected the following GPUs:", gpus)
else:
    print("No GPUs detected by TensorFlow.")
batch_size = 32
img_height = 180
img_width = 180
data_dir = pathlib.Path("./flower_tiny")
img_count = len(list(data_dir.glob("*/*.jpg")))  # 统计数据集图片数量
print(img_count)

train_ds = utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)
val_ds = utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)
# tf.print(train_ds, val_ds)
class_names = train_ds.class_names
print(class_names)

AUTOTUNE = tf.data.experimental.AUTOTUNE
# .shuffle(1000) 随机打乱数据集
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_class = len(class_names)
input_shape = (img_height, img_width, 3)
data_augmentation = Sequential(
    [

        layers.RandomFlip(
            "horizontal", 42, input_shape=input_shape
        ),
        # layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ]
)

model = Sequential(
    [
        layers.Input(shape=(180, 180, 3)),
        data_augmentation,
        layers.Rescaling(1.0 / 255),

        layers.Conv2D(16, (3, 3), padding="valid", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(32, (3, 3), padding="valid", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3, 3), padding="valid", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_class),
    ]
)

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.summary()

epochs = 24
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
)

acc = history.history['accuracy']
val_ds = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure("Acc & Loss")
plt.subplot(1, 2, 1)
plt.title("Acc")
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_ds, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.subplot(1, 2, 2)
plt.title("Loss")
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.show()


# 测试
img = PIL.Image.open("./image.png")
img = img.resize((img_height, img_width), PIL.Image.Resampling.LANCZOS)
img_array = utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(f'预测结果{class_names[tf.argmax(score)],np.max(score)}')
plt.imshow(img)
plt.show()
