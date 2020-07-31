# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 16:17:36 2020

@author: Aashu
"""


import pickle
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import contractions
import re


pickle_in = open('clf_knn.pkl', 'rb')
classifier = pickle.load(pickle_in)

pickle_in = open('Logistic_clf_cv_1.pkl', 'rb')
cv = pickle.load(pickle_in)


headlines = ["A 1962 News Report & Cautionary Tale: Why Indian Army is Wary of China’s Current 'Disengagement'"]

news = ' '.join(str(x) for x in headlines)

all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
sm = PorterStemmer()
corpus = []

news= re.sub("[^a-zA-Z]", " ", news)
news = contractions.fix(news)
news = news.lower()
news = word_tokenize(news) 
news = [sm.stem(word) for word in news if not word in set(all_stopwords)]
news = ' '.join(news)
corpus.append(news)
    
    
X = cv.transform(corpus)

index = classifier.predict(X)
    

