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
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#%%
nb = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", BernoulliNB()),
    ]
)
X_train, X_test, y_train, y_test = train_test_split(
    X_san["text"], y, test_size=0.1, random_state=69
)

start = time.time()

nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

stop = time.time()
print(stop - start)
print(f"accuracy {accuracy_score(y_pred, y_test)}")
print(classification_report(y_test, y_pred))

#%%
"""
DecisionTreeClassifier(),
MultinomialNB(),
ComplementNB(),
LogisticRegression(solver="saga"),
RidgeClassifier(solver="auto"),
SVC(),
RandomForestClassifier(),
"""
pipelines = []
for model in [BernoulliNB()]:
    pipeline = make_pipeline(TfidfVectorizer(), model)
    pipelines.append(pipeline)

#%%
berCV = GridSearchCV(
    estimator=ridge_reg,
    param_grid=params_ridge,
    scoring="neg_mean_squared_error",
    cv=folds,
)

training_time = []
for pipeline in pipelines:
    start = time.time()
    pipeline.fit(X_train, y_train)
    stop = time.time()
    training_time.append(stop - start)
