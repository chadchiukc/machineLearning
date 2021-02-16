from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import numpy as np
from nltk import NaiveBayesClassifier
import timeit
import string
import nltk
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# load data from file and separate the text and the label
def load_data(filename):
    X = []
    y = []
    with open(filename) as file:
        docs = file.read().splitlines()
        for doc in docs:
            text, label = doc.split(';')  # split the data with x and label as sep=';'
            X.append(text)
            y.append(label)
    return X, y


# create a lemma function by wordnet lemmatizer for normalizing all the tokens into lemma form
def lemmatization(text):
    wnl = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)  # tokenize the text into tokens first then lemmatize each token
    return [wnl.lemmatize(token) for token in tokens]


def training():
    Xtrain, ytrain = load_data('naturalLanguageProcessing/sentiment_analysis/train.txt')
    xtest, ytest = load_data('naturalLanguageProcessing/sentiment_analysis/val.txt')

    # initialize the vector representing the term frequency for each document
    tf_vectorizer = CountVectorizer(
        stop_words='english',  # to remove all the stop words defined by sklearn
        tokenizer=lemmatization,  # to lemmatize each token into lemma form
        lowercase=True,  # to convert all the token into lower case
        max_df=0.8,  # to ignore terms that the document frequency is higher than 80% of all documents
        min_df=6,  # to ignore terms that the count of terms is below 6
        ngram_range=(1, 2)  # to allow unigrams and bigrams to be extracted
    )

    # transform all samples into list of vectors with all the vocabs in the Xtrain(some tokens have been removed like
    # stop words or the terms with higher document frequency etc as described in the tf_vectorizer)
    X_train_tf = tf_vectorizer.fit_transform(Xtrain)

    X_test_tf = tf_vectorizer.transform(xtest)  # transform testing samples into list of vectors with vocabs in Xtrain
    nbc = MultinomialNB()  # initialize MNB
    nbc.fit(X_train_tf, ytrain)  # train the
    yhat = nbc.predict(X_test_tf)
    print(np.mean(yhat == ytest))
