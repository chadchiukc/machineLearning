import tensorflow_core.python.framework.random_seed
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tensorflow_core.python.framework.random_seed.set_seed(250)


def load_data(binary=False):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.0
    if binary:
        y_train = np.unpackbits(np.array(y_train)).reshape(-1, 8)[:, 4:]
        y_test = np.unpackbits(np.array(y_test)).reshape(-1, 8)[:, 4:]
    else:
        y_train = keras.utils.to_categorical(y_train)
        y_test = keras.utils.to_categorical(y_test)
    return x_train, y_train, x_test, y_test


def network(binary=False):
    inputs = keras.Input(shape=784)
    layer1 = keras.layers.Dense(512, activation="relu", name="layer1")(inputs)
    if binary:
        outputs = keras.layers.Dense(4, activation='relu')(layer1)
    else:
        outputs = keras.layers.Dense(10, activation="relu")(layer1)
    net = keras.Model(inputs=inputs, outputs=outputs)
    return net


def optimization(net):
    # loss = keras.losses.BinaryCrossentropy(from_logits=False)
    loss = keras.losses.MeanSquaredError()
    # loss = keras.losses.CategoricalCrossentropy(from_logits=False)
    net.compile(
        loss=loss,
        optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.8),
        metrics=['accuracy'],
    )
    return net


def train(x_train, y_train, net):
    net.fit(x_train, y_train, batch_size=32, epochs=5)
    return net


def test(x_test, y_test, net):
    net.evaluate(x_test, y_test, batch_size=32)
    return 0


def demo(binary=False):
    x_train, y_train, x_test, y_test = load_data(binary)
    net = network(binary)
    net = optimization(net)
    net = train(x_train, y_train, net)
    test(x_test, y_test, net)
    return net


def trained_model():
    _, x_train, _, x_test = load_data(binary=False)
    _, y_train, _, y_test = load_data(binary=True)
    inputs = keras.Input(shape=10)
    layer1 = keras.layers.Dense(64, activation='relu')(inputs)
    outputs = keras.layers.Dense(4, activation="relu")(layer1)
    net = keras.Model(inputs=inputs, outputs=outputs)
    net = optimization(net)
    net = train(x_train, y_train, net)
    test(x_test, y_test, net)
    return net


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data(binary=True)
    new_2 = trained_model()
    net = demo(binary=False)
    outputs = net.predict(x_train)
    y_pred = new_2.predict(outputs)
    loss = keras.losses.mean_squared_error(y_train, y_pred)
    print(np.mean(loss))