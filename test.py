from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import gensim
import numpy as np
from nltk import FreqDist, classify, NaiveBayesClassifier
import timeit

import re, string, random



def preprocess_data(filename):
    x_train = []
    y_train = []
    train = []
    wnl = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    with open(filename) as f:
        docs = f.readlines()
        for doc in docs:
            x, y = doc.split(';')
            x = word_tokenize(x)
            token_list = []
            for token, tag in pos_tag(x):
                if tag.startswith("NN"):
                    pos = 'n'
                elif tag.startswith('VB'):
                    pos = 'v'
                else:
                    pos = 'a'

                token = wnl.lemmatize(token, pos)
                token_list.append(token)
            x_train.append(token_list)
            # # pre_train = [
            #     wnl.lemmatize(w.lower()) for w in word_tokenize(x) if wnl.lemmatize(w.lower()) not in stop_words]
            # x_train.append(pos_tag(pre_train))
            # x_train.append(pos_tag([
            #     wnl.lemmatize(w.lower()) for w in word_tokenize(x) if wnl.lemmatize(w.lower()) not in stop_words]))
            y_train.append(y.replace('\n', ''))
    return x_train, y_train


def load_data(filename):
    data = []
    with open(filename) as f:
        docs = f.read().splitlines()
        for doc in docs:
            data.append(doc.split(';'))
    return data

start = timeit.default_timer()
x = load_data('naturalLanguageProcessing/sentiment_analysis/train.txt')
print(x)
# x, y = preprocess_data('train.txt')
# print(x)
# dictionary = gensim.corpora.Dictionary(x)
# print(dictionary)
# y = np.array(y)
# print(np.unique(y))
end = timeit.default_timer()
print(end - start)

