from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import numpy as np
from nltk import NaiveBayesClassifier
import timeit
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# load data from file, then tokenize all the sentence and remove stop words
def load_data_by_tokenize_lemma(filename):
    x_train = []
    y_train = []
    stop_words = stopwords.words('english')
    with open(filename) as file:
        docs = file.read().splitlines()
        for doc in docs:
            x, y = doc.split(';') # split the data with x and label as sep=';'
            x = [token.lower() for token in word_tokenize(x) if token.lower() not in stop_words]
            token_list = lemmatization(x)
            x_train.append(token_list)
            y_train.append(y)
    return np.array(x_train, dtype=object), np.array(y_train, dtype=object)


# get the vocabulary from training dataset
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


# pre-process all the datapoint in dataset by adding all the vocab into each datapoint.
def preprocess_with_vocab(x, vocab):
    train_data = []
    for sent in x:
        new_sent = {word: (word in sent) for word in vocab}
        train_data.append(new_sent)
    return train_data

# plot the confusion matrix with pre-defined classes
def confusion_matrix_plot(ytest, ypred):
    classes = ['joy', 'fear', 'love', 'anger', 'sadness', 'surprise']
    cm = confusion_matrix(ytest, ypred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot()
    plt.show()


start = timeit.default_timer()

# x, y = load_data_by_tokenize_lemma('train.txt')
# vocab = get_vocabulary(x)
# train_data = preprocess_with_vocab(x, vocab)
# train_data = np.c_[(train_data, y)] ## concentrate x with label for training the model
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

xtest, ytest = load_data_by_tokenize_lemma('train2.txt')
# classifier.show_most_informative_features()
xtest = preprocess_with_vocab(xtest, vocab)
ypred = classifier.classify_many(xtest)
confusion_matrix_plot(ytest, ypred)

end = timeit.default_timer()
print(end - start)
