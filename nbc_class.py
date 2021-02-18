from sklearn.datasets import load_iris
import numpy as np


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

    # separate the dataset into a dict with different classes
    def dataset_by_class(self):
        self.class_dict = {c: self.Xtrain[self.ytrain == c] for c in np.unique(self.ytrain)}

    # calculate the condition distribution in a dict with a key for each class and value for each feature
    def condi_distribution_by_class(self):
        self.class_condi = {k: [[column.mean(), column.std(ddof=1)] for column in np.transpose(v)] for k, v in
                            self.class_dict.items()}

    # calculate the prior distribution in a dict with a key for each class and value for the class
    def prior_distribution_by_class(self):
        self.class_prior = {i: np.log(np.count_nonzero(self.ytrain == i) / len(self.ytrain)) for i in
                            np.unique(self.ytrain)}

    # fit the model with X and y
    def fit(self, Xtrain, ytrain):
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.dataset_by_class()
        self.condi_distribution_by_class()
        self.prior_distribution_by_class()

    # calculate the normal distribution p.d.f. of a feature with log space
    def normal_pdf_w_log(self, feature, feature_mean, feature_std):
        feature_std = feature_std if feature_std != 0 else 10 ** -6  # return a small number if std is 0
        try:
            return np.log(
                np.exp(-((feature - feature_mean) / feature_std) ** 2 / 2) / (feature_std * (2 * np.pi) ** .5))
        except ValueError:
            return 10 ** -6  # return a small number if output is 0

    # predict the yhat with Xtest and only return the result with the highest prob.
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
    ytrain_hat = nbc.predict(Xtrain)
    yhat = nbc.predict(Xtest)
    train_accuracy = np.mean(ytrain_hat == ytrain)
    test_accuracy = np.mean(yhat == ytest)
    print('Train accuracy is: %.2f%%' % (train_accuracy * 100))
    print('Test accuracy is: %.2f%%' % (test_accuracy * 100))


###############################
if __name__ == '__main__':
    demo()
