# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 15:31:10 2020

@author: Aashu
"""


## Importing the library
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

## Reading the dataset
df = pd.read_csv('Combined_News_DJIA.csv')

## Load the corpus list
pickle_in = open('stem_corpus.pkl', 'rb')
corpus_stem = pickle.load(pickle_in)

## creating bag of words model
cv = CountVectorizer(ngram_range = (1,1))
X = cv.fit_transform(corpus_stem)
y = df.iloc[:, 1].values

## Spliting into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, shuffle = False)

## xgBoost model
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

## Predicting the result
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)
report_knn = classification_report(y_test, y_pred)

## Cross Validation
cross_score = cross_val_score(classifier, X, y, scoring = 'accuracy', cv = 10)
print(cross_score.mean())
print(cross_score.std())

with open('xg_rep.pkl', 'wb') as f:
    pickle.dump(report_knn, f)
    
with open('xg_clf.pkl', 'wb') as f:
    pickle.dump(classifier, f)