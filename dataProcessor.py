from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder
import re
from bertopic import BERTopic
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

nltk.download('omw-1.4')
# load data
df = pd.read_json("genreData.json")
print(df.columns)

# count vectoriser
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(","))

genres_dtm = vectorizer.fit_transform(df["genre"])

print("Some of our dps :", genres_dtm.shape[0])
print("Number of Unique genres :", genres_dtm.shape[1])

# Show all Genres
genres = vectorizer.get_feature_names()
print("Our genres are :", genres[:10])

# Create a matrix of how often our genres appear
freqs = genres_dtm.sum(axis=0).A1
res = dict(zip(genres, freqs))

print("Here is a breakdown of how foten each genre appeared in our dataset:", res)

df.groupby('genre').count()
# Data preprocessing

# some text cleaning


def clean_text(text):
    # remove backslash-apostrophe
    text = re.sub("\'", "", text)
    # remove everything except alphabets
    text = re.sub("[^a-zA-Z]", " ", text)
    # remove whitespaces
    text = ' '.join(text.split())
    # convert text to lowercase
    text = text.lower()

    return text


df.loc[:, 'blurb'] = df.loc[:, 'blurb'].apply(lambda x: clean_text(x))

# Removing stopwords from the column summary.
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# this removes stopwords


def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)


df['blurb'] = df['blurb'].apply(lambda x: remove_stopwords(x))

# Perfrom lemmatisation on summary
nltk.download('wordnet')

lemma = WordNetLemmatizer()


def lematizing(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = lemma.lemmatize(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence


df['blurb'] = df['blurb'].apply(lambda x: lematizing(x))

# Stemming on genre
stemmer = PorterStemmer()


def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence


df['blurb'] = df['blurb'].apply(lambda x: stemming(x))

# labeling each genre with a number
LE = LabelEncoder()
y = LE.fit_transform(df['genre'])
LE.inverse_transform([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


# Train test split of 80-20%
X_train, X_test, y_train, y_test = train_test_split(
    df['blurb'], y, test_size=0.2, random_state=161)
# tf-idf on blurb for both train and test x values

tfidf = TfidfVectorizer(max_df=0.8, max_features=10000)

tfidf_X_train = tfidf.fit_transform(X_train.values.astype('U'))
tfidf_X_test = tfidf.transform(X_test.values.astype('U'))

logistic = LogisticRegression()
# train logistic model
logistic.fit(tfidf_X_train, y_train)
# predict on test values
y_pred_logistic = logistic.predict(tfidf_X_test)

print('Accuracy score :', accuracy_score(y_test, y_pred_logistic))
print('Report : ')
print(classification_report(y_test, y_pred_logistic))
print(len(y_pred_logistic))

# KNN
knn = KNeighborsClassifier(n_neighbors=65)
knn.fit(tfidf_X_train, y_train)

y_pred_knn = knn.predict(tfidf_X_test)
print('Accuracy Score :', accuracy_score(y_test, y_pred_knn))
print('Report : ')
print(classification_report(y_test, y_pred_knn))

# SVM Linear classifer
svc = svm.SVC(kernel='linear').fit(tfidf_X_train, y_train)
y_pred_svc = svc.predict(tfidf_X_test)
print('Accuracy Score :', accuracy_score(y_test, y_pred_svc))
print('Report : ')
print(classification_report(y_test, y_pred_svc))
