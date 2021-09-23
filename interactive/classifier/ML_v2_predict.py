# pylint: skip-file

"""
This file classifies a new tweet or string as being about covid or not using the trained ML model. See "ML_clfv2.py".
"""

#%%
import sys, os

sys.path.append(os.path.abspath(os.path.join("..", "..", "src")))

import re
import time

import pandas as pd
import pickle
import numpy as np

import langid
from langid.langid import LanguageIdentifier, model

identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

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

from common.app import App
from common.database import Database
from common.classify import Classifier
from common.helpers import Helpers

from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

#%%
app_run = App(debug=False)
data_path = os.path.join(app_run.root_dir, "interactive", "data")
MODEL = "sgd_logreg_ridge_23092021.pkl"

#%%
# Utils funcs


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


def get_theme(row: pd.Series):
    """
    After classification by the 3 algorithms, return the majority vote.
    """

    sgd = row.loc["sgdclassifier"]
    logreg = row.loc["logisticregression"]
    ridge = row.loc["ridgeclassifier"]

    # If < 2, then at least 2 have a 0
    # same  weight for each model
    if sgd + logreg + ridge < 2:
        return 0
    return 1


def prepare_df(df, lang="en"):
    df_do = df.copy()

    # Only keep already coded tweets (6XX or "theme_hardcoded == 0")
    df_do = df_do[~df_do["topic"].isnull() | ~df_do["theme_hardcoded"].isnull()]
    df_do["text"] = df_do.apply(
        lambda r: r["old_text"] if r["text"] is None else r["text"], axis=1
    )

    # If 608 or theme_hardcoded == 0, then set new col `y` = 0
    df_do["y"] = df_do.apply(covid_classify, axis=1)

    # Make "x" col
    df_do["x_text"] = df_do["text"].progress_apply(lambda x: sanitize(x, lang))
    df_do["x_handle"] = df_do["handle"]
    df_do["x_date"] = df_do["created_at"].progress_apply(
        lambda r: r[:10].replace("-", "/"),
    )  # convert created_at to "%d/%m/%y"
    # x_date is not perfect, some are %d/%m/%y, others %y/%d/%m

    df_do["x"] = df_do.progress_apply(
        lambda r: f'{r["x_date"]} {r["x_handle"]} {r["x_text"]}', axis=1
    )

    return df_do


def prepare_row(row, lang="en"):
    row_do = row.copy()

    row_do.loc["y"] = np.nan

    # Make "x" col
    row_do.loc["x_text"] = sanitize(row_do["text"], lang)
    row_do.loc["x_handle"] = row_do["handle"]
    row_do.loc["x_date"] = row_do["created_at"][:10].replace(
        "-", "/"
    )  # convert created_at to "%d/%m/%y"
    # x_date is not perfect, some are %d/%m/%y, others %y/%d/%m

    row_do.loc["x"] = f'{row_do["x_date"]} {row_do["x_handle"]} {row_do["x_text"]}'

    return row_do


def lang_detect(txt, threshold=0.9):
    """
    Detect tweet language
    returns None if confidence lvl < threshold
    """

    if txt is None:
        return None

    txt = txt.replace("\n", " ")
    lang = identifier.classify(txt)
    if lang[1] < threshold:
        return None
    else:
        return lang[0]


def classify(row, clf):
    """
    Postprocessing classifying function.
    Allows to add post-treatment rules.
    """

    if isinstance(row, pd.Series):
        txt = row.loc["old_text"] if row.loc["text"] is None else row.loc["text"]
    else:
        txt = row

    if clf.classify(txt):
        return 1  # About covid
    else:
        if isinstance(row, pd.Series):
            return row.loc["covid_theme"]  # If already classified as covid, dont update
        return 0  # No keyword detected in str


#%%
# Predict a new tweet
# Load the fitted pipelines
loaded_pipes = pickle.load(open(os.path.join(data_path, "models", MODEL), "rb"))

#%%
# Test the loaded models
for pipe in loaded_pipes:
    res = pipe.predict(["fake tweets", "about covid"])
    print(res)

#%%
def predict_covid(tw, clf):
    """
    Classify a tweet as being about covid or not with the trained ML algorithm.

    Tweet can be either a string or a pandas.core.series.Series (dataframe row) with at least the following columns: ["handle", "created_at", "text"].
    """

    accepted_lang = ["en", "fr"]

    if isinstance(tw, str):  # is a string
        detected_lang = lang_detect(tw)
        LANG = detected_lang if detected_lang in accepted_lang else "all"

        if classify(tw, clf):
            return 1  # Covid-keyword detected
        else:
            tw = sanitize(tw, LANG)
            results = []
            for i, pipeline in enumerate(loaded_pipes):
                stepname = pipeline.steps[2][0]
                results.append(pipeline.predict([tw]))

            # Vote
            return int(bool(sum(results) > 2))

    else:  # pd.Series
        detected_lang = lang_detect(tw["text"])
        LANG = detected_lang if detected_lang in accepted_lang else "all"
        tw_do = tw.copy()

        tw_do.loc["text"] = tw["old_text"] if tw["text"] is None else tw["text"]
        tw_do = prepare_row(tw_do, lang=LANG)

        for i, pipeline in enumerate(loaded_pipes):
            stepname = pipeline.steps[2][0]

            # Predict
            tw_do.loc[stepname] = int(pipeline.predict([tw_do["x"]]))

        # Make final decision by majority vote
        tw_do.loc["covid_theme"] = get_theme(tw_do)
        # Override if keyword detected
        tw_do.loc["covid_theme"] = classify(tw_do, clf)

        return tw_do, tw_do["covid_theme"]


# TODO : add this func to classifier module

# %%
if __name__ == "__main__":
    df_other = pd.read_pickle(os.path.join(data_path, "db_other.pkl"))
    classifier = Classifier()

    tw1 = df_other.iloc[42, :]  # guess 0
    tw2 = df_other.iloc[314, :]  # 0
    tw3 = df_other.iloc[128, :]  # 0
    tw4 = df_other.iloc[888, :]  # 1
    tws = [tw1, tw2, tw3, tw4]

    str1 = df_other.iloc[32, :]["text"]  # guess 0
    str2 = df_other.iloc[33, :]["text"]  # 0
    str3 = df_other.iloc[413, :]["text"]  # 1 -> is 0 atm
    str4 = df_other.iloc[821, :]["text"]  # 1
    strs = [str1, str2, str3, str4]

    for tw in tws:
        _, cov = predict_covid(tw, classifier)
        # print(cov)

    for txt in strs:
        print(predict_covid(txt, classifier))


# %%
