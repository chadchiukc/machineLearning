This project is to use naive bayes classifier for sentiment analysis.

Before running the .py file, please make sure you have downloaded the package from nltk.
you may do it as follows:
import nltk
nltk.download()

In the root directory(i.e. lab1_homework), there is a .py file name as NLP_sentiment_analysis.py.
Make sure you have change the os dictionary that contain the NLP_sentiment_analysis.py
There are three main functions inside the python file.
1. training() is to train the model and give out the result. The model trained and the vectorizer used will also be saved while running.
2. test_prediction() is to create a prediction file from a testing file without label
3. app.run() is to act as a backend server to render html for inputting one sentence and given out the prediction.
Choose one function you want and comment out others function under __main__.

There are 4 .txt files.
1. train.txt  # for training the model with label in each sample
2. val.txt  # for validate the result with label in each sample
3. test_data.txt  # a test data for prediction
4. test_prediction.txt  # a prediction data based on the test_data predicted by trained model

There are 2 .pkl files.
1. nbc_model.pkl  # the saved trained model
2. tf_vectorizer.pkl  # the saved pre-defined vectorized with pre-defined bag-of-words

There are 2 sub-directory.
1. templates  # there is a .html file inside template for rendering
2. static  # all the assets (emoji png files) are placed inside