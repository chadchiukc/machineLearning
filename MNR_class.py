from sklearn.datasets import load_iris
import numpy as np
import math


def load_data():
    iris = load_iris()
    X, y = iris['data'], iris['target']
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


class NBC:

    def __init__(self):
        self.class_dict = None
        self.Xtrain = None
        self.ytrain = None
        self.class_condi = None
        self.class_prior = None

    ## separate the dataset into a dict with different classes
    def dataset_by_class(self):
        class_dict = {}
        for i in range(len(self.Xtrain)):
            class_value = self.ytrain[i]
            if class_value not in class_dict:
                class_dict[class_value] = []
            class_dict[class_value].append(self.Xtrain[i])
        self.class_dict = class_dict

    def condi_distribution_by_class(self):
        class_condi = {}
        for k, v in self.class_dict.items():
            class_condi[k] = [[column.mean(), column.std(ddof=1)] for column in np.transpose(v)]
        self.class_condi = class_condi

    def prior_distribution_by_class(self):
        self.class_prior = {i: math.log10(np.count_nonzero(self.ytrain == i) / len(self.ytrain)) for i in
                            np.unique(self.ytrain)}

    def fit(self, Xtrain, ytrain):
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.dataset_by_class()
        self.condi_distribution_by_class()
        self.prior_distribution_by_class()

    def normal_pdf_w_log(self, x, mean, std):
        std = std if std != 0 else 10 ** -6
        try:
            return math.log10(math.exp(-((x - mean) / std) ** 2 / 2) / (std * (2 * math.pi) ** .5))
        except ValueError:
            return 10 ** -6

    def predict(self, Xtest):
        prob_set = []
        for datapoint in Xtest:
            prob = {}
            for k, v in self.class_condi.items():
                prob[k] = self.class_prior[k]
                for i in range(len(v)):
                    mean, std = v[i]
                    prob[k] += self.normal_pdf_w_log(datapoint[i], mean, std)
            prob_set.append(prob)
        return np.array([max(v, key=v.get) for v in prob_set])


def demo():
    X, y = load_data()
    Xtrain, ytrain, Xtest, ytest = shuffle_data(X, y)

    nbc = NBC()
    nbc.fit(Xtrain, ytrain)
    yhat = nbc.predict(Xtest)
    test_accuracy = np.mean(yhat == ytest)
    print(test_accuracy)

###############################
if __name__ == '__main__':
    demo()