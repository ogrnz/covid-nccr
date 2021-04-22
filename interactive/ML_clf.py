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
        row.loc["topic"] == "608"
        or row.loc["topic"] == "608.0"
        or row.loc["theme_hardcoded"] == "0"
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


def get_theme(row: pd.Series):
    ber = row.loc["bernoullinb"]
    logreg = row.loc["logisticregression"]
    ridge = row.loc["ridgeclassifier"]

    # If < 2, then at least 2 have a 0
    # same  weight for each model
    if ber + logreg + ridge < 2:
        return 0
    return 1


#%%
# General pipeline structure
pipelines = []
for model in [LogisticRegression(), BernoulliNB(), RidgeClassifier()]:
    pipeline = make_pipeline(CountVectorizer(), TfidfTransformer(), model)
    pipelines.append(pipeline)

folds = KFold(n_splits=10, shuffle=True, random_state=31415)

# %%
dfs = [df_en, df_fr, df_other]

for df in dfs:
    # Set lang for preprocessing
    LANG = "en"
    if df["lang"].iloc[0] == "fr":
        LANG = "fr"
    elif df["lang"].iloc[0] == "other":
        LANG = "all"

    print("\n", "-" * 50)
    print(f"\nStarting for {LANG} dataset")

    # Preprocess
    # 1: keep only already coded tweets
    df_coded = df[~df["topic"].isnull() | ~df["theme_hardcoded"].isnull()].copy()
    df_test = df[df["topic"].isnull() & df["theme_hardcoded"].isnull()].copy()

    # 2: Set `y` column
    df_coded.loc[:, "y"] = df_coded.copy().progress_apply(covid_classify, axis=1)
    df_test.loc[:, "y"] = np.nan

    # 3: create new col "x" with text or oldText if text is nan
    df_coded.loc[:, "x"] = df_coded.progress_apply(
        lambda r: r["oldText"] if r["text"] is None else r["text"], axis=1
    )
    df_test.loc[:, "x"] = df_test.progress_apply(
        lambda r: r["oldText"] if r["text"] is None else r["text"], axis=1
    )

    # 4: Sanitize text
    df_coded.loc[:, "x"] = df_coded["x"].progress_apply(
        lambda t: sanitize(t, lang=LANG)
    )
    df_test.loc[:, "x"] = df_test["x"].progress_apply(lambda t: sanitize(t, lang=LANG))

    y_train = df_coded.loc[:, "y"].ravel()

    for i, pipeline in enumerate(pipelines):
        stepname = pipeline.steps[2][0]
        print(stepname)
        start = time.time()

        pipeline.fit(df_coded.loc[:, "x"], y_train)
        df_test.loc[:, stepname] = pipeline.predict(df_test.loc[:, "x"])
        df_coded.loc[:, stepname] = pipeline.predict(df_coded.loc[:, "x"])

        print(f"Computed in {time.time() - start}s")
        score = accuracy_score(df_coded.loc[:, "y"], df_coded.loc[:, stepname])
        print(f"Accuracy score (train set): {score}")

    df_test.loc[:, "covid_theme"] = np.nan
    df_test.loc[:, "covid_theme"] = df_test.progress_apply(get_theme, axis=1)
    df_coded.loc[:, "covid_theme"] = np.nan
    df_coded.loc[:, "covid_theme"] = df_coded.progress_apply(get_theme, axis=1)

    df_not = df_coded[df_coded["theme_hardcoded"] == "0"].copy()
    false_neg = 1 - (df_not["covid_theme"] == 0).sum() / len(df_not)
    print(f"\nFalse negative: {false_neg}")

    df_yes = df_coded[
        ~(df_coded["theme_hardcoded"] == "0")
        & ~(df_coded["topic"] == "608")
        & ~(df_coded["topic"] == "608.0")
    ].copy()
    false_pos = 1 - (df_yes["covid_theme"] == 1).sum() / len(df_yes)
    print(f"False positive: {false_pos}")

# %%
