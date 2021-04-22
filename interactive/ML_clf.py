"""
After investigation (see ML_clf_lab.py), the selected models are:
1. LogisticRegression
2. RidgeClassifier
3. BernoulliNB
All have more than 94% accuracy for the english and french sets and about 90% for the general `other` set.

For the final classifier, the approach is the following:
1. For each set (en, fr, other), classify each tweets with the 3
algorithms trained on the respective set
2. Final choice is made if at least 2/3 algs have the same choice
"""

#%%
import sys, os

sys.path.append(os.path.abspath(os.path.join("..", "src")))

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

# %%
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier

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
# Import the sets
df_en = pd.read_pickle("data/db_en.pkl")
df_fr = pd.read_pickle("data/db_fr.pkl")
df_other = pd.read_pickle("data/db_other.pkl")

#%%
# Preprocessing fcts


def covid_classify(row: pd.Series):
    if (
        row["topic"] == "608"
        or row["topic"] == "608.0"
        or row["theme_hardcoded"] == "0"
    ):
        return 0
    return 1


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


#%%
# General pipeline structure
pipelines = []
for model in [LogisticRegression(), BernoulliNB(), RidgeClassifier()]:
    pipeline = make_pipeline(CountVectorizer(), TfidfTransformer(), model)
    pipelines.append(pipeline)
folds = KFold(n_splits=10, shuffle=True, random_state=31415)

# %%
# Eng

# Preprocess
# 1: keep only already coded tweets
df_en_coded = df_en[~df_en["topic"].isnull() | ~df_en["theme_hardcoded"].isnull()]
df_en_test = df_en[df_en["topic"].isnull() & df_en["theme_hardcoded"].isnull()]

# 2: Set `y` column
df_en_coded["y"] = df_en_coded.copy().progress_apply(covid_classify, axis=1)
df_en_test["y"] = np.nan

# 3: create new col "x" with text or oldText if text is nan
df_en_coded["x"] = df_en_coded.progress_apply(
    lambda r: r["oldText"] if r["text"] is None else r["text"], axis=1
)
df_en_test["x"] = df_en_test.progress_apply(
    lambda r: r["oldText"] if r["text"] is None else r["text"], axis=1
)
X = df_en_coded["x"]
y = df_en_coded["y"].ravel()
print(X)
print(y)

# 4: Sanitize X
# Preprocess text
X_san = X.copy()
X_san = X_san.progress_apply(lambda t: sanitize(t, lang="en"))
print(X_san)
print(X_san[X_san.str.contains("https")])

#%%
# Get rest of english tweets
#%%
for i, pipeline in enumerate(pipelines):
    start = time.time()
    CV_scores = cross_val_score(
        pipeline, X_san, y, scoring="accuracy", cv=folds, n_jobs=-1
    )
    pipeline.predict()
    print(i)
    print(f"Mean CV accuracy: {np.mean(CV_scores)}")
    print(f"Computed in {time.time() - start}s")

# %%
