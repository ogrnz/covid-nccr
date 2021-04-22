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
from nltk.stem.snowball import SnowballStemmer

stopwords_en = nltk.corpus.stopwords.words("english")
stemmer_en = SnowballStemmer(language="english")

stopwords_fr = nltk.corpus.stopwords.words("french")
stemmer_fr = SnowballStemmer(language="french")

stopwords_all = (
    nltk.corpus.stopwords.words("english")
    + nltk.corpus.stopwords.words("french")
    + nltk.corpus.stopwords.words("german")
    + nltk.corpus.stopwords.words("italian")
)

from tqdm import tqdm

tqdm.pandas()


from common.database import Database
from common.app import App

#%%
df_en = pd.read_pickle("data/db_en.pkl")
df_fr = pd.read_pickle("data/db_fr.pkl")
df_other = pd.read_pickle("data/db_other.pkl")

""" ENG """

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
def sanitize(text, lang="en"):
    if lang == "en":
        stemmer = stemmer_en
        stopwords = stopwords_en
    elif lang == "fr":
        stemmer = stemmer_fr
        stopwords = stopwords_fr
    elif lang == "all":
        stemmer = stemmer_en
        stopwords = stopwords_all

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
        [stemmer.stem(word) for word in edited_text if word not in stopwords]
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
# Lemmatization instead of stemming?

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
from sklearn.pipeline import make_pipeline


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

#%%
# Overall pipeline

pipelines = []
for model in [LogisticRegression(), BernoulliNB(), RidgeClassifier()]:
    pipeline = make_pipeline(CountVectorizer(), TfidfTransformer(), model)
    pipelines.append(pipeline)
folds = KFold(n_splits=10, shuffle=True, random_state=31415)

# %%
train_times = []
for i, pipeline in enumerate(pipelines):
    start = time.time()
    CV_scores = cross_val_score(
        pipeline, X_san["text"], y, scoring="accuracy", cv=folds, n_jobs=-1
    )
    train_times.append(time.time() - start)
    print(i)
    print(f"Mean CV accuracy: {np.mean(CV_scores)}")
    print(f"Computed in {time.time() - start}s")

"""
df_en:
LogReg - 95%
BernoulliNB - 93%
RidgeClassifier - 94%
"""

# %%
"""FR"""

# 1: keep only already coded tweets
df_fr_coded = df_fr[~df_fr["topic"].isnull() | ~df_fr["theme_hardcoded"].isnull()]
#%%
# 2: Set `y` column
df_fr_coded["y"] = df_fr_coded.copy().progress_apply(covid_classify, axis=1)

# %%
# 3: create new col "x" with text or oldText if text is nan

df_fr_coded["x"] = df_fr_coded.apply(
    lambda r: r["oldText"] if r["text"] is None else r["text"], axis=1
)
X = df_fr_coded["x"]
y = df_fr_coded["y"].ravel()
print(X)
print(y)

# %%
# 4: Sanitize X
# Preprocess text
X_san = X.copy()
X_san = X_san.progress_apply(lambda t: sanitize(t, lang="fr"))
print(X_san)
print(X_san[X_san.str.contains("https")])  # if empty -> good
# Delete accent?

# %%
train_times = []
for i, pipeline in enumerate(pipelines):
    start = time.time()
    CV_scores = cross_val_score(
        pipeline, X_san, y, scoring="accuracy", cv=folds, n_jobs=-1
    )
    train_times.append(time.time() - start)
    print(i)
    print(f"Mean CV accuracy: {np.mean(CV_scores)}")
    print(f"Computed in {time.time() - start}s")

"""
df_fr:
LogReg - 94%
BernoulliNB - 95%
RidgeClassifier - 95%
"""

# %%
# Experiment: try with "df_other" and treat it as english+french+de+it
"""Other"""

# 1: keep only already coded tweets
df_other_coded = df_other[
    ~df_other["topic"].isnull() | ~df_other["theme_hardcoded"].isnull()
]
#%%
# 2: Set `y` column
df_other_coded["y"] = df_other_coded.copy().progress_apply(covid_classify, axis=1)

# %%
# 3: create new col "x" with text or oldText if text is nan

df_other_coded["x"] = df_other_coded.apply(
    lambda r: r["oldText"] if r["text"] is None else r["text"], axis=1
)
X = df_other_coded["x"]
y = df_other_coded["y"].ravel()
print(X)
print(y)

# %%
# 4: Sanitize X
# Preprocess text
X_san = X.copy()
X_san = X_san.progress_apply(lambda t: sanitize(t, lang="all"))
print(X_san)
print(X_san[X_san.str.contains("https")])  # if empty -> good

# %%
train_times = []
for i, pipeline in enumerate(pipelines):
    start = time.time()
    CV_scores = cross_val_score(
        pipeline, X_san, y, scoring="accuracy", cv=folds, n_jobs=-1
    )
    train_times.append(time.time() - start)
    print(i)
    print(f"Mean CV accuracy: {np.mean(CV_scores)}")
    print(f"Computed in {time.time() - start}s")

"""
df_other:
LogReg - 91%
BernoulliNB - 91%
RidgeClassifier - 94%
-> better use ridge classifier
"""

# %%
