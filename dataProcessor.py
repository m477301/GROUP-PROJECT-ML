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
from umap import UMAP
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc

nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
# load data
df = pd.read_json("genreData.json")
print(df.columns)


def crossValidGraph(Ci_range, mean_error, std_error):
    plt.errorbar(Ci_range, mean_error, yerr=std_error)
    plt.xlabel('Ci')
    plt.ylabel('Accuracy')
    plt.show()


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

stop_words = set(stopwords.words('english'))

# this removes stopwords


def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)


df['blurb'] = df['blurb'].apply(lambda x: remove_stopwords(x))

# Perfrom lemmatisation on summary


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

Ci_range = [0.1, 0.5, 1, 5, 10, 20, 50]
mean_error = []
std_error = []
bestScoreLogistic = -5
for Ci in Ci_range:
    logistic = LogisticRegression(penalty="l2", C=Ci)
    # train logistic model
    logistic.fit(tfidf_X_train, y_train)
    # predict on test values
    y_pred_logistic = logistic.predict(tfidf_X_test)
    scores = accuracy_score(y_test, y_pred_logistic)
    if (scores > bestScoreLogistic):
        print(scores)
        mostAccurateModelLogistic = logistic
        bestScoreLogistic = scores
        best_y_pred = y_pred_logistic
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())
crossValidGraph(Ci_range, mean_error, std_error)

print('Accuracy score :', bestScoreLogistic)
print('Report : ')
print(classification_report(y_test, best_y_pred))
# print(len(y_pred_logistic))

# # KNN
knn = KNeighborsClassifier(n_neighbors=65)
knn.fit(tfidf_X_train, y_train)

y_pred_knn = knn.predict(tfidf_X_test)
print('Accuracy Score :', accuracy_score(y_test, y_pred_knn))
print('Report : ')
print(classification_report(y_test, y_pred_knn))

# SVM Linear classifer
svc = svm.SVC(kernel='linear').fit(tfidf_X_train, y_train)
y_pred_svc = svc.predict(tfidf_X_test)
mean_error = []
std_error = []
bestScoreSVM = -5
for Ci in Ci_range:
    svc = svm.SVC(kernel='linear', C=Ci).fit(tfidf_X_train, y_train)
    # train logistic model
    # predict on test values
    y_pred_svm = svc.predict(tfidf_X_test)
    scores = accuracy_score(y_test, y_pred_svm)
    if (scores > bestScoreSVM):
        print(scores)
        mostAccurateModelSVM = svc
        bestScoreSVM = scores
        best_y_pred = y_pred_svm
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())
crossValidGraph(Ci_range, mean_error, std_error)
print('Accuracy Score :', bestScoreSVM)
print('Report : ')
print(classification_report(y_test, best_y_pred))


# Training model using blurb, title, author and pages
# df.loc[:, 'title'] = df.loc[:, 'title'].apply(lambda x: clean_text(x))
# df['title'] = df['title'].apply(lambda x: remove_stopwords(x))
# df['title'] = df['title'].apply(lambda x: lematizing(x))
# df['title'] = df['title'].apply(lambda x: stemming(x))
# df.loc[:, 'author'] = df.loc[:, 'author'].apply(lambda x: clean_text(x))
# df['author'] = df['author'].apply(lambda x: remove_stopwords(x))
# df['author'] = df['author'].apply(lambda x: lematizing(x))
# df['author'] = df['author'].apply(lambda x: stemming(x))


# X1 = (df['blurb'])
# tfidf_X1 = tfidf.fit_transform(X1.values.astype('U'))
# X2 = (df['title'])
# tfidf_X2 = tfidf.fit_transform(X2.values.astype('U'))
# X3 = (df['author'])
# tfidf_X3 = tfidf.fit_transform(X3.values.astype('U'))
# X4 = (df['pages'])
# tfidf_X4 = tfidf.fit_transform(X3.values.astype('U'))
# X = np.column_stack((X1, X2))
# X = np.column_stack((X, X3))
# X = np.column_stack((X, X4))
# # Train test split of 80-20%
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=161)
# print(X_train)


# # train logistic model
# logistic.fit(X_train, y_train)
# # predict on test values
# y_pred_logistic = logistic.predict(X_test)

# # Logistic with all features
# print('Accuracy score :', accuracy_score(y_test, y_pred_logistic))
# print('Report : ')
# print(classification_report(y_test, y_pred_logistic))
# print(len(y_pred_logistic))

# y_score = logistic.predict_proba(X_test)
# fpr, tpr, _ = roc_curve(y_test, y_score[:,1], pos_label=1)
# roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
# roc_auc = auc(fpr, tpr)
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.title("ROC with Logistic Regression Classifier")
# plt.show()

# # knn with all features
# k_range = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
# mean_error = []; std_error = []
# for k in k_range:
#     model = KNeighborsClassifier(n_neighbors=k)
#     temp=[]
#     kf = KFold(n_splits=5)
#     for train, test in kf.split(X):
#         model.fit(X[train], y[train])
#         ypred = model.predict(X[test])

#         temp.append(f1_score(y[test],ypred,average='micro'))
#     mean_error.append(np.array(temp).mean())
#     std_error.append(np.array(temp).std())
# plt.errorbar(k_range, mean_error, yerr = std_error)
# plt.xlabel("k"); plt.ylabel("F1 Score")
# plt.title(f"Cross-Validation with kNN")
# plt.show()

y_pred_knn = knn.predict(X_test)
print('Accuracy Score :', accuracy_score(y_test, y_pred_knn))
print('Report : ')
print(classification_report(y_test, y_pred_knn))

# y_score = knn.predict_proba(X_test)
# fpr, tpr, _ = roc_curve(y_test, y_score[:,1], pos_label=1)
# roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
# roc_auc = auc(fpr, tpr)
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.title("ROC with Logistic Regression Classifier")
# plt.show()

# Svm with al features
# svc = svm.SVC(kernel='linear').fit(X_train, y_train)
# y_pred_svc = svc.predict(X_test)
# print('Accuracy Score :', accuracy_score(y_test, y_pred_svc))
# print('Report : ')
# print(classification_report(y_test, y_pred_svc))
