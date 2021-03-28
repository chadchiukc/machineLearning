from tensorflow import keras
import numpy as np


def load_data():
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    img_size = 28  # MNIST image width and height
    col, row = np.meshgrid(np.arange(img_size), np.arange(img_size))
    coord = np.stack((col, row), axis=2).reshape(-1, 2)


load_data()