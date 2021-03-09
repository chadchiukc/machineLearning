import numpy as np
import matplotlib.pyplot as plt


a = [0.8424500226974487, 0.8871166706085205, 0.8353166580200195]
b = [0.857200026512146, 0.8932999968528748, 0.8553000092506409]


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


bar_graph('Accuracy vs regulazier', ['L1', 'L2', 'L1 and L2'], 'Accuracy', a, b)
