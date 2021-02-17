from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
import matplotlib.pyplot as plt
import joblib
from flask import Flask, render_template, request


app = Flask(__name__)


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


def confusion_matrix_plot(ytest, ypred):
    classes = ['joy', 'fear', 'love', 'anger', 'sadness', 'surprise']
    cm = confusion_matrix(ytest, ypred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot()
    plt.show()


def training():
    Xtrain, ytrain = load_data('train.txt')
    xtest, ytest = load_data('val.txt')

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
    nbc.fit(X_train_tf, ytrain)  # train the model by training data
    ypred = nbc.predict(X_test_tf)  # test the model
    confusion_matrix_plot(ytest, ypred)  # draw the confusion matrix result by comparing ytest and ypred
    print('The accuracy score: %.2f%%' % (accuracy_score(ytest, ypred) * 100))
    print('The f1_score: %.2f' % (f1_score(ytest, ypred, average='macro')))
    joblib.dump(nbc, 'nbc_model.pkl')
    joblib.dump(tf_vectorizer, 'tf_vectorizer.pkl')


def nbc_prediction(sample):
    tf_vectorizer = joblib.load('tf_vectorizer.pkl')
    nbc = joblib.load('nbc_model.pkl')
    sample_tf = tf_vectorizer.transform(sample)
    return nbc.predict(sample_tf)


if __name__ == "__main__":
    # training()
    app.run()