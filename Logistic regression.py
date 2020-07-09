# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 09:44:02 2020

@author: Aashu
"""


## Importing libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle

## Reading the Dataset
df = pd.read_csv('Combined_News_DJIA.csv')

## reading the corpus of stemmed data
pickle_in = open('stem_corpus.pkl', 'rb')
corpus = pickle.load(pickle_in)

## creating bag of words model
cv = CountVectorizer(ngram_range = (1,1))
X = cv.fit_transform(corpus)
y = df.iloc[:, 1].values

## Spliting into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, shuffle = False)

## Logistic regression model
classifier = LogisticRegression( max_iter = 1000, random_state = 0, C=0.95)
classifier.fit(X_train, y_train)

## Testing accuracy of testing dataset
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

## Cross Score Validation
cross_score = cross_val_score(classifier, X, y, scoring = 'accuracy', cv = 10)
print(cross_score.mean())

with open('Logistic_clf_1.pkl', 'wb') as f:
    pickle.dump(classifier, f)
    
with open('Logistic_clf_rep_1.pkl', 'wb') as f:
    pickle.dump(report, f)
    

with open('Logistic_clf_cv_1.pkl', 'wb') as f:
    pickle.dump(cv, f)

    


