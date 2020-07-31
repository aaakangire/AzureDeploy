# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 10:36:36 2020

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
from sklearn.ensemble import RandomForestClassifier

## Reading the dataset
df = pd.read_csv('Combined_News_DJIA.csv')

## Load the corpus list
pickle_in = open('stem_corpus.pkl', 'rb')
corpus_stem = pickle.load(pickle_in)

pickle_in = open('lamma_corpus.pkl', 'rb')
corpus_lamma = pickle.load(pickle_in)

## creating bag of words model
cv = CountVectorizer(ngram_range = (1,1))
X = cv.fit_transform(corpus_stem)
y = df.iloc[:, 1].values

## Spliting into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, shuffle = False)

## Random Forest Classifier
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

## Predicting the test results
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))

## Cross Validation
cross_score = cross_val_score(classifier, X, y, scoring = 'accuracy', cv = 10)
print(cross_score.mean())

## gridsearchCv
## Hyperparameter tuning
param_grid = [
    {
     'n_estimators' : [100, 200, 500],
     'criterion' : ['gini', 'entropy'],
     'max_features' :['sqrt', 'log2'],
     'bootstrap' : [True],
     'max_samples' : [0.1, 0.3, 0.5, 0.7, 0.9],
     'min_samples_split' : [2, 3, 4]
     }
    ]

clf = GridSearchCV(classifier, param_grid = param_grid, cv = 3,scoring = 'roc_auc', refit = True,  verbose=True, n_jobs=-1)
best_clf = clf.fit(X_train, y_train)

hyper_para = best_clf.best_params_

## Testing the hyper parameters
classifier2 = RandomForestClassifier(n_estimators = 100, criterion = 'entropy',  max_features = 'log2', max_samples = 0.3, min_samples_split = 2)
classifier2.fit(X_train, y_train)
y_pred2 = classifier2.predict(X_test)
print(confusion_matrix(y_test, y_pred2))
report = classification_report(y_test, y_pred2)
print(cross_val_score(classifier2, X, y, scoring = 'accuracy', cv = 10))

with open('Randomforest_clf_report.pkl', 'wb') as f:
    pickle.dump(report, f)
    

with open('Randomforest_clf.pkl', 'wb') as f:
    pickle.dump(classifier2, f)