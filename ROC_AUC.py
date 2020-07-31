# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:03:26 2020

@author: Aashu
"""


import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


pickle_in = open('stem_corpus.pkl', 'rb')
corpus = pickle.load(pickle_in)

df = pd.read_csv('Combined_News_DJIA.csv')

## creating bag of words model
cv = CountVectorizer(ngram_range = (2,4))
X = cv.fit_transform(corpus)
y = df.iloc[:, 1].values

## Spliting into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, shuffle = False)

## Logistic regression model
classifier2 = LogisticRegression( max_iter = 1000, random_state = 0, C=0.95)
classifier2.fit(X_train, y_train)
y_pred_log = classifier2.predict_proba(X_test)
y_pred_log = y_pred_log[:, 1]

## KNN model
classifier = KNeighborsClassifier(n_neighbors = 10)
classifier.fit(X_train, y_train)
y_pred_knn = classifier.predict_proba(X_test)
y_pred_knn = y_pred_knn[:, 1]

## Plotting roc_auc
log_fpr, log_tpr, threshold = roc_curve(y_test, y_pred_log)
auc_log = auc(log_fpr, log_tpr)

knn_fpr, knn_tpr, threshold = roc_curve(y_test, y_pred_knn)
auc_knn = auc(knn_fpr, knn_tpr)

plt.figure(figsize = (5,5), dpi = 100)
plt.plot(log_fpr, log_tpr, linestyle = '-', label = 'Logistic (auc = %0.3f)'% auc_log)
plt.plot(knn_fpr, knn_tpr, linestyle = 'dotted', label = 'KNN (auc = %0.3f)'% auc_knn)

plt.xlabel('False Positive Rate')
plt.ylabel('true Positive Rate')

plt.legend()
plt.show()

data = corpus[1499]