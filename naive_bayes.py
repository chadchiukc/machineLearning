from sklearn.datasets import load_iris
import numpy as np
import math
import random

IRIS_VAR = ['Setosa', 'Versicolour', 'Virginica']


def load_data():
    iris = load_iris()
    X, y = iris['data'], iris['target']
    # iris = np.c_[iris['data'], iris['target']]
    return X, y


def shuffle_data(X, y):
    N, D = X.shape
    Ntrain = int(0.8 * N)
    shuffler = np.random.permutation(N)
    Xtrain = X[shuffler[:Ntrain]]
    ytrain = y[shuffler[:Ntrain]]
    Xtest = X[shuffler[Ntrain:]]
    ytest = y[shuffler[Ntrain:]]
    return Xtrain, ytrain, Xtest, ytest


# class NBC:

# def __init__(self, Xtrain, ytrain, Xtest, ytest):
#     self.Xtrain = Xtrain
#     self.ytrain = ytrain
#     self.Xtest = Xtest
#     self.ytest = ytest


def dataset_by_class(X, y):
    class_dict = {}
    for i in range(len(X)):
        class_value = y[i]
        if class_value not in class_dict:
            class_dict[class_value] = []
        class_dict[class_value].append(X[i])
    return class_dict


def condi_distribution(X):
    condi_dis = [[column.mean(), column.std(ddof=1)] for column in np.transpose(X)]
    return condi_dis


def condi_distribution_by_class(X, y):
    class_dict = dataset_by_class(X, y)
    class_condi = {}
    for k, v in class_dict.items():
        class_condi[k] = condi_distribution(v)
    return class_condi


def normal_pdf_w_log(x, mean, std):
    try:
        return math.log10(math.exp(-((x - mean) / std) ** 2 / 2) / (std * (2 * math.pi) ** .5))
    except (ValueError, ZeroDivisionError):
        return 10**-6


def prior_distribution_by_class(ytrain):
    class_prior = {i: math.log10(np.count_nonzero(ytrain == i) / len(ytrain)) for i in np.unique(ytrain)}
    return class_prior


def fit(Xtrain, ytrain):
    class_condi = condi_distribution_by_class(Xtrain, ytrain)
    class_prior = prior_distribution_by_class(ytrain)
    return class_condi, class_prior


def datapoint_predict_prob(condi, prior, datapoint):
    prob = {}
    for k, v in condi.items():
        prob[k] = prior[k]
        for i in range(len(v)):
            mean, std = v[i]
            prob[k] += normal_pdf_w_log(datapoint[i], mean, std)
    return prob


def predict_prob(condi, prior, dataset):
    return [datapoint_predict_prob(condi, prior, datapoint) for datapoint in dataset]


def predict(condi, prior, dataset):
    pred = predict_prob(condi, prior, dataset)
    return np.array([max(v, key=v.get) for v in pred])


print(normal_pdf_w_log(2,3,0))

# X, y = load_data()
# c
# condi, prior = fit(Xtrain, ytrain)
# yhat = predict(condi, prior, Xtest)
# print(ytest)
# print (np.mean(yhat == ytest))


def cross_validation_split(dataset, n_folds):
    dataset_split = []
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    print(fold_size)
    for _ in range(n_folds):
        fold_tmp = []
        while len(fold_tmp) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold_tmp.append(dataset_copy.pop(index))
        dataset_split.append(fold_tmp)
    return dataset_split
