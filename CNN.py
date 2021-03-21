import os
from tensorflow import keras
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import losses
import numpy as np
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def visualize_data():
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    for i in range(10):
        plt.subplot(2, 10, i + 1)
        plt.imshow(random.choice(X_train[y_train == i]), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(i)
        plt.subplot(2, 10, i + 11)
        plt.imshow(random.choice(X_test[y_test == i]), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(i)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def load_data():
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32') / 255.0
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32') / 255.0
    # y_train = keras.utils.to_categorical(y_train)
    # y_test = keras.utils.to_categorical(y_test)
    return X_train, y_train, X_test, y_test


def network():
    inputs = keras.Input(shape=(28, 28, 1))
    layer1 = layers.Conv2D(25, (12, 12), strides=2, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, (5, 5), padding='same', activation="relu")(layer1)
    layer3 = layers.MaxPooling2D((2, 2))(layer2)
    layer4 = layers.Flatten()(layer3)
    layer5 = layers.Dense(1024, activation='relu')(layer4)
    outputs = layers.Dense(10, activation='softmax')(layer5)
    net = keras.Model(inputs=inputs, outputs=outputs)
    # print(net.summary())
    return net


def optimization(net):
    net.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'],
                lr=1e-4)


def train(X_train, y_train, X_test, y_test, net):
    history = net.fit(X_train, y_train, batch_size=50, epochs=1, validation_data=(X_test, y_test))
    return history


def demo():
    X_train, y_train, X_test, y_test = load_data()
    net = network()
    optimization(net)
    history = train(X_train, y_train, X_test, y_test, net)
    print(history)


# network()
# load_data()
# visualize_data()
if __name__ == "__main__":
    demo()
    # visualize_data()