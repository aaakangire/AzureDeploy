# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 09:41:40 2020

@author: Aashu
"""
import pickle
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


pickle_in = open('Logistic_clf_cv_1.pkl', 'rb')
cv = pickle.load(pickle_in)

pickle_in = open('Logistic_clf_1.pkl', 'rb')
log_clf = pickle.load(pickle_in)

pickle_in = open('clf_knn.pkl', 'rb')
knn_clf = pickle.load(pickle_in)

pickle_in = open('Randomforest_clf.pkl', 'rb')
rf_clf = pickle.load(pickle_in)

pickle_in = open('svm_clf_poly.pkl', 'rb')
svm_clf = pickle.load(pickle_in)

pickle_in = open('xg_clf.pkl', 'rb')
xg_clf = pickle.load(pickle_in)

pickle_in = open('stem_corpus.pkl', 'rb')
corpus = pickle.load(pickle_in)

## Reading the dataset
df = pd.read_csv('Combined_News_DJIA.csv')

## Countvectorizer
X = cv.transform(corpus)
y = df.iloc[:, 1].values

## Spliting into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, shuffle = False)

## Voting classifier hard
estimators = [('rf', rf_clf), ('xg', xg_clf), ('knn', knn_clf)]
vot_hard = VotingClassifier(estimators = estimators, voting ='hard') 
vot_hard.fit(X_train, y_train)

## Predicting the test results
y_pred = vot_hard.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

## Cross Validation
cross_score = cross_val_score(vot_hard, X, y, scoring = 'accuracy', cv = 10)
print(cross_score.mean())

with open('vot_clf.pkl', 'wb') as f:
    pickle.dump(vot_hard, f)


