import os
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def multi_layers(num_layers):
    layer = [keras.Input(shape=(28 * 28))]
    l = [layers.Dense(256, activation="relu")] * int(num_layers)
    layer.append(l)
    layer.append(layers.Dense(10))
    return layer


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.0
    return x_train, y_train, x_test, y_test


# 'relu' / LeakyReLu(0.01) / Sigmond / Tanh
# output activation: softmax /
def network(num_layers, num_neurons, activation, out_activation):
    layer = [keras.Input(shape=(28 * 28))]
    l = [layers.Dense(num_neurons, activation=activation)] * int(num_layers)
    layer.append(l)
    layer.append(layers.Dense(10, activation=out_activation))
    return keras.Sequential(layer)


def optimization(net):
    net.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.5),
        # optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.5),
        # optimizer = tf.keras.optimizers.Adam(lr=0.001),
        metrics=['accuracy'],
    )
    return net


def train(x_train, y_train, net):
    loss, acc = net.fit(x_train, y_train, batch_size=32, epochs=5)
    return net, loss, acc


def test(x_test, y_test, net):
    results = net.evaluate(x_test, y_test, batch_size=32)
    return results


def demo():
    x_train, y_train, x_test, y_test = load_data()
    net = network(5)
    optimization(net)
    net = train(x_train, y_train, net)
    test(x_test, y_test, net)


if __name__ == "__main__":
    demo()


