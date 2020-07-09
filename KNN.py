# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:46:22 2020

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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

## Reading the dataset
df = pd.read_csv('Combined_News_DJIA.csv')

## Load the corpus list
pickle_in = open('stem_corpus.pkl', 'rb')
corpus_stem = pickle.load(pickle_in)

pickle_in = open('Logistic_clf_cv_1.pkl', 'rb')
cv = pickle.load(pickle_in)

## creating bag of words model
X = cv.transform(corpus_stem)
y = df.iloc[:, 1].values

## Spliting into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, shuffle = False)

## KNN model
classifier = KNeighborsClassifier(n_neighbors = 10)
classifier.fit(X_train, y_train)

## Testing the accuracy
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)
report_knn = classification_report(y_test, y_pred)

## Cross Validation
cross_score = cross_val_score(classifier, X, y, scoring = 'accuracy', cv = 10)
print(cross_score.mean())
print(cross_score.std())

## pickle the classifier and report
with open('clf_knn.pkl', 'wb') as f:
    pickle.dump(classifier, f)
    
with open('clf_knn_rep.pkl', 'wb') as f:
    pickle.dump(report_knn, f)
    
with open('clf_knn_rep.pkl', 'wb') as f:
    pickle.dump(report_knn, f)