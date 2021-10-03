import tensorflow
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
    loss = keras.losses.MeanSquaredError()
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


def first_model(binary=False):
    x_train, y_train, x_test, y_test = load_data(binary)
    net = network(binary)
    net = optimization(net)
    net = train(x_train, y_train, net)
    test(x_test, y_test, net)
    return net


def second_model():
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


def third_model():
    x_train, y_train, x_test, y_test = load_data(binary=True)
    inputs = keras.Input(shape=784)
    layer1 = keras.layers.Dense(512, activation="relu", name="layer1")(inputs)
    layer2 = keras.layers.Dense(10, activation="relu")(layer1)
    layer3 = keras.layers.Dense(64, activation='relu')(layer2)
    outputs = keras.layers.Dense(4, activation="relu")(layer3)
    net = keras.Model(inputs=inputs, outputs=outputs)
    net = optimization(net)
    net = train(x_train, y_train, net)
    test(x_test, y_test, net)
    return net


def partA():
    net = first_model(binary=False)
    print(net.summary())
    net = first_model(binary=True)
    print(net.summary())


def partB():
    x_train, y_train, x_test, y_test = load_data(binary=True)
    new_2 = second_model()
    print(new_2.summary())
    net = first_model(binary=False)
    outputs = net.predict(x_test)
    y_pred = new_2.predict(outputs)
    loss = keras.losses.mean_squared_error(y_test, y_pred)
    print('new loss: {}'.format(np.mean(loss)))
    acc = tensorflow.metrics.binary_accuracy(y_true=y_test, y_pred=y_pred)
    print('new accuracy: {}'.format(np.mean(acc)))


def partC():
    net = third_model()
    print(net.summary())


if __name__ == "__main__":
    # partA()
    partB()
    # partC()