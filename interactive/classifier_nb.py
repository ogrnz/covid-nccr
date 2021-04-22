# pylint: skip-file
#%%
import sys, os

sys.path.append(os.path.abspath(os.path.join("..", "src")))

#%%
import re
import time

import pandas as pd
import numpy as np
import nltk
from tqdm import tqdm

tqdm.pandas()


from common.database import Database
from common.app import App

#%%
df_en = pd.read_pickle("data/db_en.pkl")
df_fr = pd.read_pickle("data/db_fr.pkl")
df_other = pd.read_pickle("data/db_other.pkl")

#%%
""" ENG """
# Drop a few cols for readability
df_en.drop(
    [
        "covid_theme",
        "created_at",
        "handle",
        "name",
        "URL",
        "type",
        "retweets",
        "favorites",
        "position",
        "frame",
    ],
    axis=1,
    inplace=True,
)

#%%
def covid_classify(row: pd.Series):
    if (
        row["topic"] == "608"
        or row["topic"] == "608.0"
        or row["theme_hardcoded"] == "0"
    ):
        return 0
    return 1


#%%
# Only keep already coded tweets (6XX or "theme_hardcoded == 0")
df_en = df_en[~df_en["topic"].isnull() | ~df_en["theme_hardcoded"].isnull()]

# If 608 or theme_hardcoded == 0, then set new col `y` = 0
df_en["y"] = df_en.apply(covid_classify, axis=1)

# %%
X = pd.DataFrame({"text": []})
X["text"] = df_en.apply(
    lambda r: r["oldText"] if r["text"] is None else r["text"], axis=1
)
y = df_en["y"].ravel()
print(X)
print(y)

#%%
from nltk.stem.snowball import SnowballStemmer

stopwords = nltk.corpus.stopwords.words("english")
snowball_stemmer = SnowballStemmer(language="english")


def sanitize(text):
    edited_text = re.sub(
        "  ", " ", text
    )  # replace double whitespace with single whitespace
    edited_text = re.sub(
        "https?:\/\/(www\.)?t(witter)?\.com?\/[a-zA-Z0-9_]+", "", edited_text
    )  # remove URL
    edited_text = re.sub(
        "https", "", edited_text
    )  # remove `https` for uncompleted tweets
    edited_text = edited_text.split(" ")  # split the sentence into array of strings
    edited_text = " ".join(
        [char for char in edited_text if char != ""]
    )  # remove any empty string from text
    edited_text = edited_text.lower()  # lowercase
    edited_text = re.sub("\d+", "", edited_text)  # Removing numerics
    edited_text = re.split(
        "\W+", edited_text
    )  # spliting based on whitespace or whitespaces
    edited_text = " ".join(
        [snowball_stemmer.stem(word) for word in edited_text if word not in stopwords]
    )  # Snowball Stemmer
    return edited_text


tmp_tweet = "We proved that women’s active participation in a peace process can make a significant difference. -- Nobel Laureate @LeymahRGbowee on the important role of women for peace.\xa0https://t.co/XHnRXGPlQk\xa0via @AfricaRenewal\xa0https://t.co/1XH5JbBegt\xa0Feb 01, 2020\xa0'"
sanitize(tmp_tweet)

# %%
# Preprocess text
X_san = X.copy()
X_san["text"] = X_san["text"].progress_apply(sanitize)
print(X_san)
print(X_san[X_san["text"].str.contains("https")])  # if empty -> good

# Remove rare and frequent words?

# %%
# Scikit time
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

#%%
"""
BerNB
~93% accuracy
"""
nb_berNB = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", BernoulliNB()),
    ]
)

start = time.time()
folds = KFold(n_splits=10, shuffle=True, random_state=31415)
CV_berNB = cross_val_score(
    nb_berNB, X_san["text"], y, scoring="accuracy", cv=folds, n_jobs=-1
)
print(f"Time {time.time() - start}")
print(f"Mean CV accuracy: {np.mean(CV_berNB)}")

# print(f"accuracy {accuracy_score(y_pred, y_test)}")
# print(classification_report(y_test, y_pred))

#%%
"""
SVC
- not scalable for that amount of data
BUT ~96% accuracy.. for 14min runtime
"""
nb_svc = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", SVC()),
    ]
)

start = time.time()
folds = KFold(n_splits=10, shuffle=True, random_state=31415)
CV_svc = cross_val_score(
    nb_svc, X_san["text"], y, scoring="accuracy", cv=folds, n_jobs=-1
)
print(f"Time {time.time() - start}")
print(f"Mean CV accuracy: {np.mean(CV_svc)}")

#%%
"""
DecisionTreeClassifier
~87% accuracy
"""
nb_dectree = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", DecisionTreeClassifier(max_features="auto")),
    ]
)

start = time.time()
folds = KFold(n_splits=50, shuffle=True, random_state=31415)
CVd_dectree = cross_val_score(
    nb_dectree, X_san["text"], y, scoring="accuracy", cv=folds, n_jobs=-1
)
print(f"Time {time.time() - start}")
print(f"Mean CV accuracy: {np.mean(CVd_dectree)}")

#%%
"""
MultinomialNB
~88% accuracy
"""
nb_multiNB = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", MultinomialNB(alpha=0.7)),
    ]
)

start = time.time()
folds = KFold(n_splits=10, shuffle=True, random_state=31415)
CV_multiNB = cross_val_score(
    nb_multiNB, X_san["text"], y, scoring="accuracy", cv=folds, n_jobs=-1
)
print(f"Time {time.time() - start}")
print(f"Mean CV accuracy: {np.mean(CV_multiNB)}")

#%%
"""
ComplementNB
~89% accuracy
"""
nb_compNB = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", ComplementNB(alpha=0.8)),
    ]
)

start = time.time()
folds = KFold(n_splits=10, shuffle=True, random_state=31415)
CV_compNB = cross_val_score(
    nb_compNB, X_san["text"], y, scoring="accuracy", cv=folds, n_jobs=-1
)
print(f"Time {time.time() - start}")
print(f"Mean CV accuracy: {np.mean(CV_compNB)}")

#%%
"""
LogisticRegression
~95% accuracy for different penalty and solver
3s l2 penalty
"""
nb_logreg = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", LogisticRegression()),
    ]
)

start = time.time()
folds = KFold(n_splits=10, shuffle=True, random_state=31415)
CV_logreg = cross_val_score(
    nb_logreg, X_san["text"], y, scoring="accuracy", cv=folds, n_jobs=-1
)
print(f"Time {time.time() - start}")
print(f"Mean CV accuracy: {np.mean(CV_logreg)}")

#%%
"""
RidgeClassifier
~94%
3s
"""
nb_ridge = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", RidgeClassifier()),
    ]
)

start = time.time()
folds = KFold(n_splits=10, shuffle=True, random_state=31415)
CV_ridge = cross_val_score(
    nb_ridge, X_san["text"], y, scoring="accuracy", cv=folds, n_jobs=-1
)
print(f"Time {time.time() - start}")
print(f"Mean CV accuracy: {np.mean(CV_ridge)}")

#%%
"""
kNN
~87% best
"""
folds = KFold(n_splits=5, shuffle=True, random_state=31415)
nb_knn = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("knn", KNeighborsClassifier()),
    ]
)
pipeline = GridSearchCV(
    nb_knn,
    {
        "knn__n_neighbors": [25, 27, 30, 32, 35],
        "knn__weights": ["distance"],
    },
    cv=folds,
    n_jobs=-1,
)


start = time.time()
pipeline.fit(X_san["text"], y)
res = pipeline.cv_results_
print(f"Time {time.time() - start}")
print(pd.DataFrame(res))


# %%
"""
BernoulliNB - 93%
LogReg - 95%
RidgeClassifier - 94%
for df_en dataset
"""

""" FR """
