# import nltk
# from nltk import word_tokenize
# from nltk.corpus import stopwords
#
# doc1 = "The housekeeping service was HORRIBLE The halls were very noisy as well."
# doc2 = "If you want a quiet complex at night do not go here. The complex staff will not take action against noisy guests."
# doc3 = "Great restaurants and bars on site, all the staff are really friendly, everywhere is kept clean and tidy."
# doc4 = "Location is the best, staff very helpful and friendly, the hotel is very well maintained and very clean."
# #
# # stop_words = stopwords.words('english')  # store the stop words from nltk
# # porter = nltk.PorterStemmer()
# #
# # for index, doc in enumerate([doc1, doc2, doc3, doc4], 1):
# #     tokens = word_tokenize(doc.replace('.', '').replace(',', ''))  # tokenize and remove ',' & '.'
# #     tokens = [token.lower() for token in tokens if token.lower() not in stop_words]  # remove stop words & lower case
# #     tokens = [porter.stem(token) for token in tokens]  # stemming
# #     print('Doc ID {}: {}.'.format(index, tokens))
#
#
# import nltk
# import math
# import string
# from nltk.corpus import stopwords
# from collections import Counter
# from nltk.stem.porter import *
#
#
# def n_containing(word, count_list): # get the number of documents containing term 'word'
#     return sum(1 for count in count_list if word in count)
#
#
# def tf(word, count):    # calculate the tf of each word
#     return count[word] / sum(count.values())
#
#
# def idf(word, count_list):  # calculate the idf of each word
#     return math.log(len(count_list) / (1 + n_containing(word, count_list)))
#
#
#
# def tfidf(word, count, count_list): # calculate the tf-idf
#     return tf(word, count) * idf(word, count_list)
#
#
# def stem_tokens(tokens, stemmer):   #
#     stemmed = []
#     for item in tokens:
#         stemmed.append(stemmer.stem(item))  # get the stem of word 'item'
#
#     return stemmed
#
#
# def get_tokens(text):   # get tokens of current document
#     lower = text.lower()  # ignore case
#     remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)  # get punctuations
#     no_punctuation = lower.translate(remove_punctuation_map)  # # remove punctuations
#     tokens = nltk.word_tokenize(no_punctuation)  # translate current documents into token list
#     return tokens
#
#
# def myCount(wordlist):  # get the counts of each word in current document
#     term_counter = Counter()
#     for word in wordlist:
#         if word not in term_counter:
#             term_counter[word] = 1.0
#         else:
#             term_counter[word] += 1.0
#
#     return term_counter
#
#
# def count_term(text):
#     tokens = get_tokens(text)  # Tokenization of current document
#     filtered = [w for w in tokens if not w in stopwords.words('english')]  # filter stopwords
#     stemmer = nltk.PorterStemmer()  # build the stemmer
#     stemmed = stem_tokens(filtered, stemmer)  # stem the words in current documents
#     counter = nltk.Counter(stemmed)  # get the counter number of each word
#
#     return counter
#
#
# if __name__ == '__main__':
#     texts = [doc1, doc2, doc3, doc4]  # concrete three documents into a list
#     countlist = []  # to record the words No. in each docment
#     for text in texts:  # each documents
#         countlist.append(count_term(text))  # call count_term to calculate the counts of each word
#     for i, count in enumerate(countlist):  # for each document
#         print("In document id {}".format(i + 1))
#         scores = {word: tfidf(word, count, countlist) for word in
#                   count}  # calculate the  TF-IDF scores of each word and store it in scores
#         sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)  # sorted the scores in descending order
#         for word, score in sorted_words:  # print the top 5 words
#             print("\t{}, TF-IDF: {}".format(word, round(score, 5)))

from tensorflow import keras
from tensorflow.keras import layers

# model = keras.Sequential()
# model.add(Conv2D(32, (3, 3), padding='same',input_shape=(128, 128, 3)))
# model.add(Activation('relu'))
# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))


def network():
    model = keras.Sequential()
    model.add(keras.Input(shape=(28, 28, 1)))
    model.add(layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    print(model.summary())


network()


def network2():
    model = keras.Sequential()
    model.add(keras.Input(shape=(28, 28, 1)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    print(model.summary())

# network2()
# import pandas as pd
# import matplotlib.pyplot as plt
#
# df = pd.read_csv('covtype.data.gz', compression='gzip', header=None)
# # df.corr().hist()
# df = df.iloc[:, 10:54]
# print(df)
# a = [1,2,3]
# b = [2,3,4]
# print(a-b)