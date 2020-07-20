## Import Libraries
from flask import Flask, render_template, request, redirect
import contractions
import re
import pickle
from nltk.stem.porter import PorterStemmer
import bs4
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen

## Load classifier
pickle_in = open('clf_knn.pkl', 'rb')
classifier = pickle.load(pickle_in)

## Load bag of words
pickle_in = open('Logistic_clf_cv_1.pkl', 'rb')
cv = pickle.load(pickle_in)

## Load stopwords
pickle_in = open('stopwords.pkl', 'rb')
all_stopwords = pickle.load(pickle_in)

## Function for output index trend
def trend(headlines):
    news = ' '.join(str(x) for x in headlines)
    sm = PorterStemmer()
    corpus = []
    news= re.sub("[^a-zA-Z]", " ", news)
    news = contractions.fix(news)
    news = news.lower()
    news = news.split()
    news = [sm.stem(word) for word in news if not word in set(all_stopwords)]
    news = ' '.join(news)
    corpus.append(news)
    X = cv.transform(corpus)
    index = classifier.predict(X)
    return index

app = Flask(__name__)

headlines = []

## Function for scrapping News Headlines
def scrap_news():
    url="https://news.google.com/news/rss"
    Client=urlopen(url)
    xml_page=Client.read()
    Client.close() 
    soup_page=soup(xml_page,"xml")
    news_list=soup_page.findAll("item")
    for news in news_list:
        headlines.append(news.title.text) 



## Homepage 
@app.route('/')
@app.route('/home', methods = ['GET', 'POST'])
def home():
    if request.method == 'POST':
        scrap_news()
        return redirect('/home')
    else: 
        return render_template('index.html')
  
## Predict Index
@app.route('/home/post')
def post():
    index = trend(headlines)
    return render_template('index.html', index = index, news = headlines)

## Clear the list
@app.route('/home/refresh')
def refresh():
    headlines.clear()
    return render_template('index.html')
    

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8000)