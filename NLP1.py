import pandas as pd   # Great for tables (google spreadsheets, microsoft excel, csv). 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import spacy
import wordcloud
import os # Good for navigating your computer's files 
import sys

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spacy.lang.en.stop_words import STOP_WORDS
nltk.download('wordnet')
nltk.download('punkt')

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
!python -m spacy download en_core_web_md
import en_core_web_md
import gdown
import sklearn.metrics as metrics


gdown.download('https://drive.google.com/uc?id=1u0tnEF2Q1a7H_gUEH-ZB3ATx02w8dF4p', 'yelp_final.csv', True)
data_file  = 'yelp_final.csv'
yelp.drop(labels=['business_id','user_id'],inplace=True,axis=1)
yelp = yelp[yelp.stars != 4]

def is_good_review(stars):
    if stars == 5:  
        return True
    else:
        return False

# Change the stars field to either be 'good' or 'bad'.
yelp['is_good_review'] = yelp['stars'].apply(is_good_review)

X = yelp['text']
y = yelp['is_good_review']
bow_transformer = CountVectorizer(analyzer=tokenize, max_features=800).fit(X)
a = bow_transformer.transform(X)
X = a


logistic_model = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
logistic_model.fit(X_train,y_train)


y_pred = logistic_model.predict(X_test)
preds = y_pred

# Get the confusion matrix.
cm = confusion_matrix(y_test, preds)

# Get TP, FP, TN, and FN rates.
TP = cm[0][0]
TN = cm[1][1]
FP = cm[0][1]
FN = cm[1][0]

accuracy = (TP + TN) / (TP + TN + FP + FN)
#accuracy = metrics.accuracy_score(y_test, preds)

print("The accuracy of the model is " + str(accuracy*100) + "%")
