from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import numpy as np
from nltk import NaiveBayesClassifier
import timeit
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def preprocess_data_from_file(filename):
    x_train = []
    y_train = []
    stop_words = stopwords.words('english')
    with open(filename) as f:
        docs = f.read().splitlines()
        for doc in docs:
            x, y = doc.split(';')
            x = [token.lower() for token in word_tokenize(x) if token.lower() not in stop_words]
            token_list = lemmatization(x)
            x_train.append(token_list)
            y_train.append(y)
    return np.array(x_train, dtype=object), np.array(y_train, dtype=object)


def get_vocabulary(x):
    return set(token for sent in x for token in sent)


def lemmatization(tokenlist):
    wnl = WordNetLemmatizer()
    token_list = []
    for token, tag in pos_tag(tokenlist):
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        token = wnl.lemmatize(token, pos)
        token_list.append(token)
    return token_list


def pretraining(x, y, vocab):
    train_data = []
    for sent in x:
        new_sent = {word: (word in sent) for word in vocab}
        train_data.append(new_sent)
    train_data = np.c_[(train_data, y)]
    return train_data


def pretraining_wo_y(x, vocab):
    train_data = []
    for sent in x:
        new_sent = {word: (word in sent) for word in vocab}
        train_data.append(new_sent)
    return train_data


def confusion_matrix_plot(ytest, ypred):
    classes = ['joy', 'fear', 'love', 'anger', 'sadness', 'surprise']
    cm = confusion_matrix(ytest, ypred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot()
    plt.show()


start = timeit.default_timer()

# x, y = preprocess_data_from_file('train.txt')
# vocab = get_vocabulary(x)
# train_data = pretraining(x, y, vocab)
# classifier = NaiveBayesClassifier.train(train_data)
# f = open('nbc.pickle', 'wb')
# pickle.dump(classifier, f)
# f.close()
# f = open('vocab.pickle', 'wb')
# pickle.dump(vocab, f)
# f.close()

f = open('nbc.pickle', 'rb')
classifier = pickle.load(f)
f.close()
f = open('vocab.pickle', 'rb')
vocab = pickle.load(f)
f.close()

xtest, ytest = preprocess_data_from_file('train2.txt')
# classifier.show_most_informative_features()
xtest = pretraining_wo_y(xtest, vocab)
ypred = classifier.classify_many(xtest)
confusion_matrix_plot(ytest, ypred)

end = timeit.default_timer()
print(end - start)

