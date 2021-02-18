from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import joblib
from flask import Flask, render_template, request
import numpy as np


app = Flask(__name__)


# act as a backend server to predict the emotion of one sentence
@app.route('/', methods=['GET', 'POST'])
def index():
    pred = []
    if request.method == 'POST':
        sent = request.form['sent']
        pred = nbc_prediction([sent])
        pred = ''.join(pred)
    return render_template('index.html', pred=pred)


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
    tokens = word_tokenize(text)  # tokenize the text into tokens first then lemmatize each token
    return [wnl.lemmatize(token) for token in tokens]


# plot the confusion matrix based on ytest and ypred
def confusion_matrix_plot(ytest, ypred):
    classes = ['joy', 'fear', 'love', 'anger', 'sadness', 'surprise']
    cm = confusion_matrix(ytest, ypred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot()
    plt.show()


# print out the precision, recall and f1score for each label
def metrics_result(ytest, ypred):
    y_precision_recall_fscore = precision_recall_fscore_support(ytest, ypred)
    metric_name = ['precision', 'recall', 'f1score']
    label_name = np.unique(ytest)
    for j in range(np.shape(y_precision_recall_fscore)[0] - 1):
        for i in range(label_name.size):
            print('The %s for %s is: %.2f' % (metric_name[j], label_name[i], y_precision_recall_fscore[j][i]))


# train the model and draw all the necessary results
def training():
    Xtrain, ytrain = load_data('lab1_homework/train.txt')
    xtest, ytest = load_data('lab1_homework/val.txt')

    # initialize the vector representing the term frequency for each document
    tf_vectorizer = CountVectorizer(
        stop_words='english',  # to remove all the stop words defined by sklearn
        tokenizer=lemmatization,  # to lemmatize each token into lemma form
        lowercase=True,  # to convert all the token into lower case
        max_df=0.8,  # to ignore terms that the document frequency is higher than 80% of all documents
        min_df=6,  # to ignore terms that the count of terms is below 6
        ngram_range=(1, 2)  # to allow both unigrams and bigrams to be extracted
    )

    # transform all samples into list of vectors with all the vocabs in the Xtrain(some tokens have been removed like
    # stop words or the terms with higher document frequency etc as described in the tf_vectorizer)
    X_train_tf = tf_vectorizer.fit_transform(Xtrain)
    X_test_tf = tf_vectorizer.transform(xtest)  # transform testing samples into list of vectors with vocabs in Xtrain
    nbc = MultinomialNB()  # initialize MNB
    nbc.fit(X_train_tf, ytrain)  # train the model by training data
    ypred = nbc.predict(X_test_tf)  # test the model
    confusion_matrix_plot(ytest, ypred)  # draw the confusion matrix result by comparing ytest and ypred
    metrics_result(ytest, ypred)  # print the metrics results for each label
    print('The overall accuracy score: %.2f%%' % (accuracy_score(ytest, ypred) * 100))

    # calculate f1score based on averaging f1score of each label
    print('The overall f1_score: %.4f' % (f1_score(ytest, ypred, average='macro')))

    # save the model for later use
    joblib.dump(nbc, 'lab1_homework/nbc_model.pkl')
    joblib.dump(tf_vectorizer, 'lab1_homework/tf_vectorizer.pkl')


# used for batch dataset prediction by loading the model trained before
def nbc_prediction(X):
    tf_vectorizer = joblib.load('lab1_homework/tf_vectorizer.pkl')
    nbc = joblib.load('lab1_homework/nbc_model.pkl')
    X_tf = tf_vectorizer.transform(X)
    return nbc.predict(X_tf)


# to create a prediction file from a file with X
def test_prediction(filename, new_filename):
    with open(filename) as file:
        docs = file.read().splitlines()
        Xtest = [doc for doc in docs]
    tf_vectorizer = joblib.load('lab1_homework/tf_vectorizer.pkl')
    nbc = joblib.load('lab1_homework/nbc_model.pkl')
    X_tf = tf_vectorizer.transform(Xtest)
    ypred = nbc.predict(X_tf)
    with open(new_filename, 'w+') as f:
        for y in ypred:
            f.write(y + '\n')


# below are three major functions:
# 1. training() is to train the model
# 2. test_prediction() is to create a prediction file from a testing file without label
# 3. app.run() is to act as a backend server to render html for one sentence prediction
# choose one function you want and comment out others function
if __name__ == "__main__":
    # training()
    # test_prediction('test_data.txt', 'test_prediction.txt')
    app.run()
