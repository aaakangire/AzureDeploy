# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:23:08 2020

@author: Aashu
"""

## Importing the library
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import contractions
import pickle

## Reading the Dataset
df = pd.read_csv('Combined_News_DJIA.csv')

## Changing the Index
headlines = []
Index = [int(i) for i in range(27)]
df.columns = Index
for row in range(0,len(df.index)):
    headlines.append(' '.join(str(x) for x in df.iloc[row,2:]))
    
## Adding new column to the dataset
df['headlines'] = headlines

## Cleaning of the data and append to stemmed corpus
corpus_stem = []
corpus_lamma = []
sm = PorterStemmer()
lm = WordNetLemmatizer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
for i in range(0, len(df.index)):
        review = re.sub("[^a-zA-Z]", " ", df['headlines'][i])
        review = contractions.fix(review)
        review = review.lower()
        review = review.split() 
        review_stem = [sm.stem(word) for word in review if not word in set(all_stopwords)]
        review_lamma = [lm.lemmatize(word) for word in review if not word in set(all_stopwords)]
        review_stem = [word for word in review_stem if len(word) > 2]
        review_lamma = [word for word in review_lamma if len(word) > 2]
        review_stem = ' '.join(review_stem)
        review_lamma = ' '.join(review_lamma)
        corpus_stem.append(review_stem)
        corpus_lamma.append(review_lamma)

## Pickling of cleaned corpus for further use
with open('lamma_corpus.pkl', 'wb') as f:
    pickle.dump(corpus_lamma, f)

with open('stem_corpus.pkl', 'wb') as f2:
    pickle.dump(corpus_stem, f2)
    
with open('stopwords.pkl', 'wb') as f2:
    pickle.dump(all_stopwords, f2)