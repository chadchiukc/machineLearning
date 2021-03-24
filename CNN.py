import os
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import losses
import tensorflow as tf
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
    history = net.fit(X_train, y_train, batch_size=50, epochs=2, validation_data=(X_test, y_test))
    return history


def custom_train(X_train, y_train, X_test, y_test, net):
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=60000).batch(batch_size=50)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_dataset = val_dataset.batch(batch_size=50)

    epochs = 1
    for epoch in range(epochs):
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = net(x_batch_train, training=True)
                train_loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(train_loss_value, net.trainable_weights)
            optimizer.apply_gradients(zip(grads, net.trainable_weights))
            train_acc_metric.update_state(y_batch_train, logits)
            if step % 100 == 0:
                print(
                    "Training loss at iteration %d: %.4f"
                    % (epoch*1200 + step, float(train_loss_value))
                )
                train_acc = train_acc_metric.result()
                print("Training acc: %.4f" % (float(train_acc),))
            if epoch * 1200 + step == 5000:
                train_acc_metric.reset_states()
                for x_batch_val, y_batch_val in val_dataset:
                    val_logits = net(x_batch_val, training=False)
                    val_loss_value = loss_fn(y_batch_train, logits)
                    val_acc_metric.update_state(y_batch_val, val_logits)
                val_acc = val_acc_metric.result()
                val_acc_metric.reset_states()
                print("Validation loss: %.4f" % (float(val_loss_value),))
                print("Validation acc: %.4f" % (float(val_acc),))
                break
    # net.save('mnist_cnn')


def visualize_filters(net):
    filters = net.layers[1].get_weights()[0]
    index = 1
    fig = plt.figure()
    fig.set_size_inches(18, 18)
    for i in range(25):
        plt.subplot(5, 5, index)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(filters[:, :, :, i].squeeze(), cmap='gray')
        index += 1
    plt.show()


# def visualize_patches():

def demo():
    X_train, y_train, X_test, y_test = load_data()
    # net = network()
    net = keras.models.load_model('mnist_cnn', compile=False)
    # visualize_filters(net)
    # optimization(net)
    # history = train(X_train, y_train, X_test, y_test, net)
    # custom_train(X_train, y_train, X_test, y_test, net)
    net = keras.Model(inputs=net.inputs, outputs=net.layers[1].output)
    # print(net.summary())
    # print(X_test[0].shape)
    test = X_test[0]
    test = np.expand_dims(test, axis=0)
    # for i in X_test:
    #     feature_map = net.predict(i)
    #     print(feature_map)
    # plt.imshow(feature_maps[0, :, :, 0])
    # plt.show()
    # print(X_test[0].shape)
    # print(net.layers[0].output.shape)
    filter, _ = net.layers[1].get_weights()
    print(filter[:,:,:,1])
    test = tf.image.extract_patches(images=test, sizes=[1,12,12,1], strides=[1,2,2,1], rates=[1,1,1,1],padding='VALID')
    print(test.shape)



if __name__ == "__main__":
    demo()
    # visualize_data()