from tensorflow import keras
from tensorflow.keras import layers
import os
import tensorflow as tf

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.0
    return x_train, y_train, x_test, y_test


def network():
    inputs = keras.Input(shape=784)
    layer1 = keras.layers.Dense(512, activation="relu", name="layer1")(inputs)
    layer2 = keras.layers.Dense(256, activation="relu", name="layer2")(layer1)
    layer3 = keras.layers.Dense(128, activation="relu", name="layer3")(layer2)
    outputs = keras.layers.Dense(10, activation="softmax")(layer3)
    # outputs = keras.layers.Dropout(.3)(outputs)
    net = keras.Model(inputs=inputs, outputs=outputs)
    return net


def optimization(net):
    net.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.5),
        # optimizer = tf.keras.optimizers.Adam(lr=0.001),
        metrics=['accuracy'],
    )
    return net


def train(x_train, y_train, net):
    net.fit(x_train, y_train, batch_size=32, epochs=10)
    return net


def test(x_test, y_test, net):
    net.evaluate(x_test, y_test, batch_size=32)
    return 0


def demo():
    x_train, y_train, x_test, y_test = load_data()
    net = network()
    optimization(net)
    net = train(x_train, y_train, net)
    test(x_test, y_test, net)


if __name__ == "__main__":
    demo()
