from tensorflow import keras
import numpy as np
from tensorflow.keras import backend as K
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

    A = D_half_norm.dot(A).dot(D_half_norm).astype('float32')
    return X_train, y_train, X_test, y_test, A


class GCN_1(tf.keras.layers.Layer):
    def __init__(self, num_features, A, **kwargs):
        super(GCN_1, self).__init__(**kwargs)
        self.num_features = num_features
        self.A = A

    def build(self, input_shape):
        super(GCN_1, self).build(input_shape)
        self.W = self.add_weight("W", shape=(input_shape[-1], self.num_features), trainable=True)
        # self.bias = self.add_weight("bias", shape=(self.num_features, ))

    def call(self, x):
        output = tf.matmul(x, self.W)
        print(type(x))
        output = tf.matmul(self.A, output)
        print(np.shape(x))
        print(np.shape(self.W))
        print(np.shape(output))
        print(np.shape(self.A))



        # return [self.A, np.maximum(0, self.A.dot(x).dot(self.W) + self.bias)]
        # return tf.tuple([self.A, tf.keras.activations.relu(output + self.bias)])
        return tf.tuple([self.A, tf.keras.activations.relu(output)])


class GCN(tf.keras.layers.Layer):
    def __init__(self, num_features, **kwargs):
        super(GCN, self).__init__(**kwargs)
        self.num_features = num_features

    def build(self, input_shape):
        super(GCN, self).build(input_shape)
        self.W = self.add_weight("W", shape=(input_shape[1][-1], self.num_features))
        # self.bias = self.add_weight("bias", shape=self.num_features)

    def call(self, x):
        A = x[0]
        X = x[1]
        output = tf.matmul(X, self.W)
        output = tf.matmul(A, output)

        # relu function
        return tf.tuple([A, tf.keras.activations.relu(output)])


class DiffPool(keras.layers.Layer):
    def __init__(self, num_features, max_clusters, **kwargs):
        super(DiffPool, self).__init__(**kwargs)
        self.max_clusters = max_clusters
        self.num_features = num_features

    def build(self, input_shape):
        super(DiffPool, self).build(input_shape)
        self.assignment_matrix = GCN(num_features=self.max_clusters)
        self.embedding_matrix = GCN(num_features=self.num_features)

    def call(self, x):
        A = x[0]
        X = x[1]

        (_, S) = self.assignment_matrix(x)
        (_, Z) = self.embedding_matrix(x)
        S = tf.keras.activations.softmax(S, axis=1)  # softmax is applied on assignment matrix
        S_T = tf.transpose(S)
        new_X = tf.matmul(S_T, Z)
        new_A = tf.matmul(S_T, tf.matmul(A, S))
        return tf.tuple([new_A, new_X])


def network(A):
    inputs = keras.Input(shape=784, )
    gcn1 = GCN_1(256, A=A, name='gcn1')(inputs)
    diff_pool_1 = DiffPool(128, 256, name='diff_pool_1')(gcn1)
    gcn2 = GCN(128, name='gcn2')(diff_pool_1)
    diff_pool_2 = DiffPool(128, 64, name='diff_pool_2')(gcn2)
    gcn3 = GCN(128, name='gcn3')(diff_pool_2)
    diff_pool_3 = DiffPool(128, 1, name='diff_pool_3')(gcn3)
    _, output = diff_pool_3
    outputs = keras.layers.Dense(10, activation="softmax")(output)
    net = keras.Model(inputs=inputs, outputs=outputs)
    print(net.summary())
    return net


def optimization(net):
    net.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        # optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.5),
        optimizer = tf.keras.optimizers.Adam(lr=0.001),
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
    net = network(A)
    optimization(net)
    net = train(X_train, y_train, net)
    test(X_train, y_test, net)


demo()
