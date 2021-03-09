import os
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.0
    return x_train, y_train, x_test, y_test


# [keras.layers.LeakyReLU(alpha=0.01), 'relu', 'sigmoid', 'tanh']
# output activation: softmax /
def network(num_layers, num_neurons, activation, out_activation, dropout, regularizer):
    inputs = keras.Input(shape=784)
    hidden_layer = keras.layers.Dense(num_neurons, activation=activation)(inputs)
    for _ in range(1, num_layers):
        hidden_layer = keras.layers.Dense(num_neurons, activation=activation)(hidden_layer)
    outputs = keras.layers.Dense(10, activation=out_activation, kernel_regularizer=regularizer)(hidden_layer)
    # outputs = keras.layers.Dense(10,)(hidden_layer)
    outputs = keras.layers.Dropout(dropout)(outputs)
    net = keras.Model(inputs=inputs, outputs=outputs)
    return net


def optimization(net, num, learning_rate, momentum):
    if num == 0:
        loss_func = keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    elif num == 1:
        loss_func = keras.losses.MeanAbsoluteError(),
    else:
        loss_func = keras.losses.MeanSquaredError(),

    net.compile(
        # optimizer=keras.optimizers.SGD(lr=learning_rate),
        optimizer=keras.optimizers.SGD(lr=learning_rate, momentum=momentum),
        loss=loss_func,
        # optimizer = tf.keras.optimizers.Adam(lr=0.001),
        metrics=['accuracy'],
    )
    return net


def train(x_train, y_train, net, batch_size, epochs):
    history = net.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    return net, history


def test(x_test, y_test, net, batch_size):
    result = net.evaluate(x_test, y_test, batch_size=batch_size)
    return result


def show_graph(title, xlabel, ylabel, train_data, test_data, y_axis):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(y_axis, train_data, label='train data')
    plt.plot(y_axis, test_data, label='test data')
    plt.legend()
    plt.show()


def bar_graph(title, xlabel, ylabel, train_data, test_data):
    N = len(xlabel)
    ind = np.arange(N)
    width = 0.35
    plt.bar(ind, train_data, width, label='Train data')
    plt.bar(ind + width, test_data, width, label='Test data')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(ind + width / 2, xlabel)
    plt.legend(loc='upper right')
    plt.show()


def demo():
    train_data = []
    test_data = []
    # y_label = [keras.layers.LeakyReLU(alpha=0.01), 'relu', 'sigmoid', 'tanh']
    # y_label = ['softmax', 'sigmoid', 'relu']
    # y_label = ['0.1', '0.01', '0.001', '0.0001', '0.00001']
    # y_label = ['0.0', '0.2', '0.4', '0.6', '0.8', 1]
    # y_label = [1, 8, 16, 32, 64, 128, 256]
    # y_label = [1, 2, 3, 4, 5, 6, 7, 8, 12, 24]
    # y_label = [0, 0.2, 0.4, 0.6, 0.8, 0.999999]
    y_label = ['l1', 'l2', 'l1_l2']
    x_train, y_train, x_test, y_test = load_data()
    for i in y_label:
        net = network(5, 256, 'relu', 'softmax', 0, i)
        optimization(net, 0, 0.001, 0)
        net, train_history = train(x_train, y_train, net, 32, 5)
        result = test(x_test, y_test, net, 32)
        train_data.append(train_history.history['accuracy'][-1])
        test_data.append(result[-1])
    # show_graph('Accuracy vs dropout rate', 'dropout rate', 'Accuracy', train_data, test_data,
    #            y_label)
    print(train_data)
    print(test_data)


if __name__ == "__main__":
    demo()
