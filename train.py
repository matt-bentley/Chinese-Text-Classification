import data_helpers

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

import numpy, textblob, pandas
# from keras.preprocessing import text, sequence
# from keras import layers, models, optimizers

print("Loading data...")
trainDF = data_helpers.load_data_and_labels()
print('loaded ' + str(len(trainDF)) + ' rows')

# split the dataset into training and validation datasets 75/25
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['class'],test_size=0.2, random_state=42)

print('Train count: ' + str(len(train_x)))
print('Evaluation count: ' + str(len(valid_x)))

print('')

# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

print('Processing Tf-Idf char vectors...')

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,6))
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x)
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)

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

accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("LR, CharLevel Vectors: ", accuracy)

print('')
print('Training SGD SVM...')

accuracy = train_model(SGDClassifier(loss='log', penalty='elasticnet', alpha=1e-6, max_iter=5, random_state=42), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("LR, CharLevel Vectors: ", accuracy)

print('')
print('Training SVC SVM...')

accuracy = train_model(LinearSVC(random_state=111, loss='hinge',C=1.3), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("LR, CharLevel Vectors: ", accuracy)