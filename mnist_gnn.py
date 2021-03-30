from tensorflow import keras
import numpy as np
import scipy
import tensorflow as tf


def load_data():
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    img_size = 28  # MNIST image width and height

    # Reshape the training image to node with 1 dimensional feature vector with normalized value
    X_train = X_train.reshape(-1, img_size ** 2).astype('float32') / 255.0
    X_test = X_test.reshape(-1, img_size ** 2).astype('float32') / 255.0

    # calculate the Adjacency matrix
    A = np.zeros((img_size ** 2, img_size ** 2), dtype=int)
    for i in range(img_size ** 2):
        for j in range(img_size ** 2):
            if abs(i - j) == 1 or abs(i - j) == img_size or i == j:
                A[i][j] = 1

    # calculate the Degree matrix
    D = np.zeros((img_size ** 2, img_size ** 2), dtype=int)
    for i in range(img_size ** 2):
        for j in range(img_size ** 2):
            if i == j:
                D[i][j] = np.sum(A[i][:], axis=0)
    D_half_norm = np.power(D, -0.5, where=D != 0)

    A = D_half_norm.dot(A).dot(D_half_norm)
    return X_train, y_train, X_test, y_test, A


class GCN(tf.keras.layers.Layer):
    def __init__(self, num_features, **kwargs):
        super(GCN, self).__init__(**kwargs)
        self.num_features = num_features

    def build(self, input_shape):
        super(GCN, self).build(input_shape)
        self.W = self.add_weight("W", shape=(input_shape[1][-1], self.num_features))
        self.bias = self.add_weight("bias", shape=self.num_features)

    def call(self, x):
        A = x[0]
        X = x[1]

        # relu function
        return [A, np.maximum(0, self.A.dot(X).dot(self.W) + self.bias)]


class DiffPool(keras.layers.Layer):
    def __init__(self, num_features, max_clusters, **kwargs):
        super(DiffPool, self).__init__(**kwargs)
        self.max_clusters = max_clusters
        self.num_features = num_features

    def build(self, input_shape):
        super(DiffPool, self).build(input_shape)
        self.assignment_matrix = GCN(num_outputs=self.max_clusters)
        self.embedding_matrix = GCN(num_outputs=self.num_features)

    def call(self, x):
        A = x[0]
        X = x[1]

        (_, S) = self.assignment_matrix(x)
        (_, Z) = self.embedding_matrix(x)

        S = tf.keras.activations.softmax(S, axis=1)  # softmax is applied on assignment matrix

        new_X = S.T.dot(Z)
        new_A = S.T.dot(A).dot(S)
        return [new_A, new_X]


def network():
    inputs = keras.Input(shape=784,)
    gcn_1 = GCN(256)(inputs)
    diff_pool_1 = DiffPool(128, 256)(gcn_1)
    gcn_2 = GCN(128)(diff_pool_1)
    diff_pool_2 = DiffPool(128, 64)(gcn_2)
    gcn_3 = GCN(128)(diff_pool_2)
    diff_pool_3 = DiffPool(128, 1)(gcn_3)
    _, output = diff_pool_3
    outputs = keras.layers.Dense(10, activation="softmax")(output)
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
    return


def demo():
    X_train, y_train, X_test, y_test, A = load_data()
    net = network()
    optimization(net)
    net = train(X_train, y_train, net)
    test(X_train, y_test, net)


demo()


