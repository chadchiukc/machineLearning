import os
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.datasets import mnist

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

# Sequential API (Very convenient, not very flexible)
# model = keras.Sequential(
#     [
#         keras.Input(shape=(28 * 28)),
#         layers.Dense(512, activation="relu"),
#         layers.Dense(256, activation="relu"),
#         layers.Dense(10),
#     ]
# )
# print(model.summary())
import sys
# model = keras.Sequential()
# model.add(keras.Input(shape=(784)))
# model.add(layers.Dense(784, activation="relu"))
# model.add(layers.Dense(256, activation="relu", name="my_layer"))
# model.add(layers.Dense(10))
# print(model.summary())
sys.exit()

model = keras.Sequential()
model.add(keras.Input(shpae=(64, 64, 1)))
model.add(Conv2D())
layer1 = layers.Conv2D(25, (5, 5), padding="same ,strides=1, activation="relu")(inputs)
layer4 = layers.Flatten()(layer3)
outputs = layers.Dense(1, activation='softmax')(layer5)
net = keras.Model(inputs=inputs, outputs=outputs)
# print(net.summary())
sys.exit()

# Functional API (A bit more flexible)
inputs = keras.Input(shape=(784))
x = layers.Dense(512, activation="relu", name="first_layer")(inputs)
x = layers.Dense(256, activation="relu", name="second_layer")(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)