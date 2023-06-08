import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X = np.concatenate((train_X, cv2.bitwise_not(train_X)))
train_y = np.concatenate((train_y, train_y))
test_X = np.concatenate((test_X, cv2.bitwise_not(test_X)))
test_y = np.concatenate((test_y, test_y))

train_X = train_X / 255.0
test_X = test_X / 255.0

train_X = train_X.astype("float32")
test_X = test_X.astype("float32")


# plt.imshow(train_X[0])
# plt.show()

tf.random.set_seed(42)

model = tf.keras.Sequential(
    [
        layers.Conv2D(
            filters=10, kernel_size=3, activation="relu", input_shape=(28, 28, 1)
        ),
        layers.Conv2D(10, 3, activation="relu"),
        layers.MaxPool2D(),
        layers.Conv2D(10, 3, activation="relu"),
        layers.Conv2D(10, 3, activation="relu"),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"],
)

model.fit(train_X, train_y, epochs=10, validation_data=(test_X, test_y))

model.save("models/saved_model.h5")
