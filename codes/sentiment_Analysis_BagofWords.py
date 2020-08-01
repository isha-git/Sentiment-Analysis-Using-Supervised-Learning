# Sentiment Analysis using Bag of Words Features and Support Vector Machine (SVM) and Logistic Regression Classification
# https://github.com/isha-git/Sentiment-Analysis-Using-Supervised-Learning

import numpy as np
import pandas as pd
import string
import re
import unicodedata

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer 
from nltk.tokenize import word_tokenize

from bs4 import BeautifulSoup
np.set_printoptions(precision =  2, linewidth = 80)


# importing dataset
url = 'https://raw.githubusercontent.com/dipanjanS/text-analytics-with-python/master/Old-First-Edition/source_code/Ch07_Semantic_and_Sentiment_Analysis/movie_reviews.csv'
data = pd.read_csv(url,sep=",") # use sep="," for coma separation. 
data.describe()

# visualizing dataset
print(data.head())

reviews = np.array(data['review'])
sentiments = np.array(data['sentiment'])

# considering 35000 samples for training and 150000 for testing
train_reviews = reviews[:35000]
train_sentiments = sentiments[:35000]
test_reviews = reviews[35000:]
test_sentiments = sentiments[35000:]

# -----------------------------------PRE-PROCESSING TEXT----------------------------------------------------------------
# remove HTML tags
def clean_text(text):
    # remove hyperlinks
    text_hyper = re.sub(r'http?:\/\/.*[\r\n]*', '', text)
    
    # remove hashtags (only remocing # sign from the text)
    text_hash = re.sub(r'#', '', text_hyper)
    
    return text_hash

# change accented characters to ASCII characters
def remove_accented(text):
    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3 
        pass

    text = unicodedata.normalize('NFD', text)           .encode('ascii', 'ignore')           .decode("utf-8")

    return str(text)

# change to lowercase
def text_lowercase(text):
    text_lower = text.lower()
    return text_lower

# remove punctuation
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

# stemming
def stem_words(text):
    stemmer = PorterStemmer()
    word_tokens = word_tokenize(text)
    text_stemmed = [stemmer.stem(word) for word in word_tokens]
    return text_stemmed

def remove_numbers(text): 
    result = re.sub(r'\d+', '', text) 
    return result

def tokenize_text(text):
    tokens = nltk.word_tokenize(text) 
    tokens = [token.strip() for token in tokens]
    return tokens

# remove stop words
def remove_stopwords(text):
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words.remove('no')
    stop_words.remove('but')
    stop_words.remove('not')

    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    filtered_text = ' '.join(filtered_tokens)

    return filtered_text

def text_preprocess(text):
    text_clean = clean_text(text)
    text_ascii = remove_accented(text_clean)
    text_punct = remove_punctuation(text_ascii)
    text_num = remove_numbers(text_punct)
    text_lower = text_lowercase(text_num)
    text_stop = remove_stopwords(text_lower)
    text_stem = stem_words(text_stop)

    return text_stem

# -----------------------------------PRE-PROCESSED TEXT-----------------------------------------------------------------

def gatherClassificationMetrics(testCategories, predictedCategories):
    accuracy = accuracy_score(testCategories, predictedCategories)
    metrics_report = classification_report(testCategories, predictedCategories)

    print("Accuracy rate: " + str(round(accuracy, 2)) + "\n")
    print(metrics_report)


normalized = []
for text in reviews:
    soup = BeautifulSoup(text)
    soupify = soup.get_text()

    normalized.append(str(text_preprocess(soupify)))

norm_train_reviews = normalized[:35000]
norm_test_reviews = normalized[35000:]


# feature engineering using Bag of Words
cv = CountVectorizer(binary = False, min_df = 0.0, max_df = 1.0)
cv_train_features = cv.fit_transform(norm_train_reviews)
cv_test_features = cv.transform(norm_test_reviews)


# classifying using Support Vector Machine (SVM)
svm = SGDClassifier(loss = 'hinge', max_iter = 100).fit(cv_train_features, train_sentiments)
svm.score(cv_test_features, test_sentiments, sample_weight=None)
gatherClassificationMetrics(test_sentiments, svm.predict((cv_test_features)))


# classifying using Logistic Regression
clf = LogisticRegression(random_state=0).fit(cv_train_features, train_sentiments)
clf.score(cv_test_features, test_sentiments)
gatherClassificationMetrics(test_sentiments, clf.predict((cv_test_features)))