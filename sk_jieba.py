import data_helpers

from time import time
import sys
import os
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import jieba
import jieba.posseg as pseg


from sklearn import feature_extraction, model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print("Loading data...")
trainDF = data_helpers.load_data_and_labels()
print('loaded ' + str(len(trainDF)) + ' rows')

# split the dataset into training and validation datasets 75/25
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['class'],test_size=0.2, random_state=42)

print('Train count: ' + str(len(train_x)))
print('Evaluation count: ' + str(len(valid_x)))

x = []

tokenized_corpus = []
for text in train_x:
    line = " ".join(jieba.cut(text))
    tokenized_corpus.append(line)
    x.append(line)

tokenized_test_corpus = []
for text in valid_x:
    line = " ".join(jieba.cut(text))
    tokenized_test_corpus.append(line)
    x.append(line)


print('Processing Count vectors...')

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(x)

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(tokenized_corpus)
xvalid_count =  count_vect.transform(tokenized_test_corpus)

print('Processing Tf-Idf vectors...')

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')
tfidf_vect.fit(x)
xtrain_tfidf =  tfidf_vect.transform(tokenized_corpus)
xvalid_tfidf =  tfidf_vect.transform(tokenized_test_corpus)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,2))
tfidf_vect_ngram.fit(x)
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(tokenized_corpus)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(tokenized_test_corpus)

def train_model(classifier, feature_vector_train, labels, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, labels)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)


print('')
print('Training Logistic Regression...')

# Linear Classifier on Count Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
print("LR, Count Vectors: ", accuracy)

# Linear Classifier on Word Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
print("LR, WordLevel TF-IDF: ", accuracy)

# Linear Classifier on Ngram Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("LR, N-Gram Vectors: ", accuracy)

print('')
print('Training SGD SVM...')

# Linear Classifier on Count Vectors
accuracy = train_model(SGDClassifier(loss='log', penalty='elasticnet', alpha=1e-6, max_iter=5, random_state=42), xtrain_count, train_y, xvalid_count)
print("SGD SV, Count Vectors: ", accuracy)

# Linear Classifier on Word Level TF IDF Vectors
accuracy = train_model(SGDClassifier(loss='log', penalty='elasticnet', alpha=1e-6, max_iter=5, random_state=42), xtrain_tfidf, train_y, xvalid_tfidf)
print("SGD SV, WordLevel TF-IDF: ", accuracy)

# Linear Classifier on Ngram Level TF IDF Vectors
accuracy = train_model(SGDClassifier(loss='log', penalty='elasticnet', alpha=1e-6, max_iter=5, random_state=42), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("SGD SV, N-Gram Vectors: ", accuracy)

print('')
print('Training SVC SVM...')

# Linear Classifier on Count Vectors
accuracy = train_model(LinearSVC(random_state=111, loss='hinge',C=1.3), xtrain_count, train_y, xvalid_count)
print("LR, Count Vectors: ", accuracy)

# Linear Classifier on Word Level TF IDF Vectors
accuracy = train_model(LinearSVC(random_state=111, loss='hinge',C=1.3), xtrain_tfidf, train_y, xvalid_tfidf)
print("LR, WordLevel TF-IDF: ", accuracy)

# Linear Classifier on Ngram Level TF IDF Vectors
accuracy = train_model(LinearSVC(random_state=111, loss='hinge',C=1.3), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("LR, N-Gram Vectors: ", accuracy)