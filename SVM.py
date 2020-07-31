# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 14:32:52 2020

@author: Aashu
"""


import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

## Reading the dataset
df = pd.read_csv('Combined_News_DJIA.csv')

pickle_in = open('stem_corpus.pkl', 'rb')
corpus = pickle.load(pickle_in)

## creating bag of words model
cv = CountVectorizer(ngram_range = (1,1))
X = cv.fit_transform(corpus)
y = df.iloc[:, 1].values

## Spliting into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 50)

## SVM Kernel model
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

## Testing accuracy of testing dataset
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))

## Testing accuracy of the training dataste
y_pred = classifier.predict(X_train)
print(confusion_matrix(y_train, y_pred))
print(accuracy_score(y_train, y_pred))


## Cross Validation
cross_score = cross_val_score(classifier, X, y, scoring = 'accuracy', cv = 10)
print(cross_score.mean())

## Hyperparameter tuning
param_grid = [
    {
     'C' : np.logspace(0, 1, 20),
    'kernel' : ['linear', 'rbf', 'sigmoid'],
    'gamma' : ['scale'], 
    'decision_function_shape' :['ovo', 'ovr'],
     }]

param_grid1 = [
    {
    'C' : np.logspace(0, 1, 20),
    'kernel' : ['poly'],
    'gamma' : ['scale', 'auto'], 
    'decision_function_shape' :['ovo', 'ovr'],
    'degree' : [3, 4]
     }]

clf = GridSearchCV(classifier, param_grid = param_grid, cv = 3,scoring = 'roc_auc', refit = True,  verbose=True, n_jobs=-1)
best_clf = clf.fit(X_train, y_train)

## Getting Hyper parameter for grid 1
hyper_para = best_clf.best_params_

## Testing the result on the hyper parameter 1
classifier2 = SVC(kernel = 'linear', decision_function_shape = 'ovo', gamma = 'scale', C = 1.0)
classifier2.fit(X_train, y_train)
y_pred2 = classifier2.predict(X_test)
print(confusion_matrix(y_test, y_pred2))
print(accuracy_score(y_test, y_pred2))
clf_rep_SVM_lin = classification_report(y_test, y_pred2)
print(cross_val_score(classifier2, X, y, scoring = 'accuracy', cv = 7).mean())

## Another grid search
clf2 = GridSearchCV(classifier, param_grid = param_grid1, cv = 3, scoring = 'roc_auc', refit = True,  verbose=True, n_jobs=-1)
best_clf2 = clf2.fit(X_train, y_train)

## Getting Hyper parameter for grid 1
hyper_para2 = best_clf2.best_params_

## Testing result on hyperparameter grid 2
classifier3 = SVC(kernel = 'poly', decision_function_shape = 'ovo', gamma = 'scale', C =1.8329807108324356, degree = 4)
classifier3.fit(X_train, y_train)
y_pred3 = classifier3.predict(X_test)
print(confusion_matrix(y_test, y_pred3))
print(accuracy_score(y_test, y_pred3))
clf_rep_SVM_poly = classification_report(y_test, y_pred3)
cross_val_score(classifier3, X, y, scoring = 'accuracy', cv = 7)

with open('SVM_clf_poly.pkl', 'wb') as f:
    pickle.dump(classifier3, f)
    
with open('SVM_clf_lin.pkl', 'wb') as f:
    pickle.dump(classifier2, f)
    
with open('SVM_rep_poly.pkl', 'wb') as f:
    pickle.dump(clf_rep_SVM_poly, f)
    
with open('SVM_lin_poly.pkl', 'wb') as f:
    pickle.dump(clf_rep_SVM_lin, f)
    