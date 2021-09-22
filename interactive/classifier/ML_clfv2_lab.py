# pylint: skip-file

"""
(v2) Experiment with ML classifier.
This is a second version with the same-ish algos, but with added handle and date fields as predictors.

3 predictors:
x_test = tweet text
x_handle = tweet user
x_date = tweet created_at %d/%m/%y
"""

#%%
import sys, os

sys.path.append(os.path.abspath(os.path.join("..", "..", "src")))

import re
import time
from datetime import datetime

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

from common.app import App
from common.database import Database
from common.helpers import Helpers

# %%
# Scikit time
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
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
app = App()
data_path = os.path.join(app.root_dir, "interactive", "data")

#%%
df_en = pd.read_pickle(os.path.join(data_path, "db_en.pkl"))
df_fr = pd.read_pickle(os.path.join(data_path, "db_fr.pkl"))
df_other = pd.read_pickle(os.path.join(data_path, "db_other.pkl"))

#%%
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


tmp_tweet = "We proved that women’s active participation in a peace process can make a significant difference. -- Nobel Laureate @LeymahRGbowee on the important role of women for peace.\xa0https://t.co/XHnRXGPlQk\xa0via @AfricaRenewal\xa0https://t.co/1XH5JbBegt\xa0Feb 01, 2020\xa0'"
sanitize(tmp_tweet)

""" ENG """
#%%
# Only keep already coded tweets (6XX or "theme_hardcoded == 0")
df_en = df_en[~df_en["topic"].isnull() | ~df_en["theme_hardcoded"].isnull()].copy()
df_en["text"] = df_en.apply(
    lambda r: r["old_text"] if r["text"] is None else r["text"], axis=1
)

# If 608 or theme_hardcoded == 0, then set new col `y` = 0
df_en["y"] = df_en.apply(covid_classify, axis=1)

# %%
# Preprocess text
df_en_san = df_en.copy()

df_en_san["x_text"] = df_en_san["text"].progress_apply(sanitize)
df_en_san["x_handle"] = df_en_san["handle"]
df_en_san["x_date"] = df_en_san["created_at"].progress_apply(
    lambda r: r[:10].replace("-", "/"),
)  # convert created_at to "%d/%m/%y"
# x_date is not perfect, some are %d/%m/%y, others %y/%d/%m

df_en_san["x"] = df_en_san.progress_apply(
    lambda r: f'{r["x_date"]} {r["x_handle"]} {r["x_text"]}', axis=1
)


#%%
# Drop columns that are not interesting
cols_to_drop = [col for col in df_en_san.columns if col not in ["tweet_id", "y", "x"]]
df_en_san = df_en_san.drop(columns=cols_to_drop, axis=1)


print(df_en_san)
print(df_en_san[df_en_san["x"].str.contains("https")])  # if empty -> good
# Remove rare and frequent words?
# Lemmatization instead of stemming?

#%%
# 5. train/test split to have an accurate score
X = df_en_san.drop(columns=["tweet_id", "y"], axis=1)["x"]
y = df_en_san.loc[:, "y"]
# X = np.array(X)  # transform X to numpy array
# y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=31415,
    test_size=0.3,
    shuffle=True,
)


#%%
"""
BerNB
~90% accuracy
4.8s
"""
nb_berNB = Pipeline(
    [
        ("vect", CountVectorizer(preprocessor=None)),
        ("tfidf", TfidfTransformer()),
        ("clf", BernoulliNB()),
    ]
)

start = time.time()
folds = KFold(n_splits=10, shuffle=True, random_state=31415)
CV_berNB = cross_val_score(
    nb_berNB, X_train, y_train, scoring="accuracy", cv=folds, n_jobs=-1
)

nb_berNB.fit(X_train, y_train)
y_pred = nb_berNB.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(f"Time {time.time() - start}")
print(f"Mean CV accuracy: {np.mean(CV_berNB)}")
print(f"Test accuracy: {score}")


#%%
"""
SGDClassifier
# Try with other hyperparams
# first try 92% accuracy with:
array([{'clf__alpha': 0.0001, 'clf__eta0': 1.7143, 'clf__learning_rate': 'adaptive', 'clf__loss': 'hinge'}]
2min
"""
nb_sgd = Pipeline(
    [
        ("vect", CountVectorizer(preprocessor=None)),
        ("tfidf", TfidfTransformer()),
        ("clf", SGDClassifier(random_state=31415)),
    ]
)

start = time.time()
folds = KFold(n_splits=10, shuffle=True, random_state=31415)
pipeline = GridSearchCV(
    nb_sgd,
    {
        "clf__loss": [
            "hinge",
        ],
        "clf__alpha": [0.0001],
        "clf__learning_rate": ["adaptive"],
        "clf__eta0": np.linspace(0.0001, 2, 50),
    },
    scoring="accuracy",
    cv=folds,
    verbose=1,
    n_jobs=-1,
)

pipeline.fit(X_train, y_train)
res = pipeline.cv_results_
print(f"Time {time.time() - start}")
print(pd.DataFrame(res))

#%%
"""
DecisionTreeClassifier
~82% accuracy
14s
"""
nb_dectree = Pipeline(
    [
        ("vect", CountVectorizer(preprocessor=None)),
        ("tfidf", TfidfTransformer()),
        ("clf", DecisionTreeClassifier(max_features="auto")),
    ]
)

start = time.time()
folds = KFold(n_splits=50, shuffle=True, random_state=31415)
CV_dectree = cross_val_score(
    nb_dectree, X_train, y_train, scoring="accuracy", cv=folds, n_jobs=-1
)

nb_dectree.fit(X_train, y_train)
y_pred = nb_dectree.predict(X_test)
score = accuracy_score(y_test, y_pred)

print(f"Time {time.time() - start}")
print(f"Mean CV accuracy: {np.mean(CV_dectree)}")
print(f"Test accuracy: {score}")

#%%
"""
MultinomialNB
~84% accuracy
7s
"""
nb_multiNB = Pipeline(
    [
        ("vect", CountVectorizer(preprocessor=None)),
        ("tfidf", TfidfTransformer()),
        ("clf", MultinomialNB(alpha=0.7)),
    ]
)

start = time.time()
folds = KFold(n_splits=50, shuffle=True, random_state=31415)
CV_multiNB = cross_val_score(
    nb_multiNB, X_train, y_train, scoring="accuracy", cv=folds, n_jobs=-1
)

nb_multiNB.fit(X_train, y_train)
y_pred = nb_multiNB.predict(X_test)
score = accuracy_score(y_test, y_pred)

print(f"Time {time.time() - start}")
print(f"Mean CV accuracy: {np.mean(CV_multiNB)}")
print(f"Test accuracy: {score}")

#%%
"""
ComplementNB
~87% accuracy
3s
"""
nb_compNB = Pipeline(
    [
        ("vect", CountVectorizer(preprocessor=None)),
        ("tfidf", TfidfTransformer()),
        ("clf", ComplementNB(alpha=0.8)),
    ]
)

start = time.time()
folds = KFold(n_splits=20, shuffle=True, random_state=31415)
CV_compNB = cross_val_score(
    nb_compNB, X_train["x"], y_train, scoring="accuracy", cv=folds, n_jobs=-1
)

nb_compNB.fit(X_train["x"], y_train)
y_pred = nb_compNB.predict(X_test["x"])
score = accuracy_score(y_test, y_pred)

print(f"Time {time.time() - start}")
print(f"Mean CV accuracy: {np.mean(CV_compNB)}")
print(f"Test accuracy: {score}")

#%%
"""
LogisticRegression
~91% accuracy
{'clf__C': 1.60002, 'clf__l1_ratio': 0.75, 'clf__penalty': 'elasticnet', 'clf__solver': 'saga'}
6.5min
"""
nb_logreg = Pipeline(
    [
        ("vect", CountVectorizer(preprocessor=None)),
        ("tfidf", TfidfTransformer()),
        ("clf", LogisticRegression(random_state=31415)),
    ]
)

start = time.time()
folds = KFold(n_splits=6, shuffle=True, random_state=31415)
pipeline = GridSearchCV(
    nb_logreg,
    {
        "clf__penalty": ["l1", "l2", "elasticnet", "none"],
        "clf__C": np.linspace(0.0001, 2, 6),
        "clf__solver": ["saga"],
        "clf__l1_ratio": np.linspace(0, 1, 5),
    },
    scoring="accuracy",
    cv=folds,
    verbose=1,
    n_jobs=-1,
)

pipeline.fit(X_train["x"], y_train)
res = pipeline.cv_results_
resdf = pd.DataFrame(res)
print(f"Time {time.time() - start}")
print(resdf)
print(resdf[resdf["rank_test_score"] == 1])
#%%
"""
RidgeClassifier
~93%
{'clf__alpha': 1.0, 'clf__solver': 'sparse_cg'}
40s
"""
nb_ridge = Pipeline(
    [
        ("vect", CountVectorizer(preprocessor=None)),
        ("tfidf", TfidfTransformer()),
        ("clf", RidgeClassifier(random_state=31415)),
    ]
)

start = time.time()
folds = KFold(n_splits=5, shuffle=True, random_state=31415)
pipeline = GridSearchCV(
    nb_ridge,
    {
        "clf__alpha": np.linspace(0.0001, 1, 10),
        "clf__solver": ["svd", "lsqr", "sparse_cg", "sag", "saga"],
    },
    scoring="accuracy",
    cv=folds,
    verbose=1,
    n_jobs=-1,
)

pipeline.fit(X_train, y_train)
res = pipeline.cv_results_
resdf = pd.DataFrame(res)
print(f"Time {time.time() - start}")
print(resdf)
print(resdf[resdf["rank_test_score"] == 1])

#%%
"""
kNN
~87% best

"""
folds = KFold(n_splits=5, shuffle=True, random_state=31415)
nb_knn = Pipeline(
    [
        ("vect", CountVectorizer(preprocessor=None)),
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

params_sgd = {
    "sgdclassifier__eta0": 1.7143,
    "sgdclassifier__learning_rate": "adaptive",
    "sgdclassifier__loss": "hinge",
}
params_logreg = {
    "logisticregression__C": 1.6,
    "logisticregression__l1_ratio": 0.75,
    "logisticregression__penalty": "elasticnet",
    "logisticregression__solver": "saga",
}
params_ridge = {"ridgeclassifier__solver": "sparse_cg"}
optimal_params = {
    "sgdclassifier": params_sgd,
    "logisticregression": params_logreg,
    "ridgeclassifier": params_ridge,
}

pipelines = []
for i, model in enumerate(
    [
        SGDClassifier(random_state=31415),
        LogisticRegression(random_state=31415),
        RidgeClassifier(),
    ]
):
    pipeline = make_pipeline(
        CountVectorizer(preprocessor=None), TfidfTransformer(), model
    )
    pipelines.append(pipeline)
    pipeline.set_params(**optimal_params[pipeline.steps[2][0]])

folds = KFold(n_splits=10, shuffle=True, random_state=31415)

#%%
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
    df_do["x"] = df_do["text"].progress_apply(lambda x: sanitize(x, lang))
    print(df_do[df_do["x"].str.contains("https")])  # if empty -> good

    return df_do


#%%
df_en_ready = prepare_df(df_en, lang="en")

#%%
# Train/test split to have an accurate score
X_train, X_test, y_train, y_test = train_test_split(
    df_en_ready.drop("y", axis=1),
    df_en_ready.loc[:, "y"],
    random_state=31415,
    test_size=0.2,
    shuffle=True,
)


# %%
for i, pipeline in enumerate(pipelines):
    start = time.time()
    CV_scores = cross_val_score(
        pipeline, X_train["x"], y_train, scoring="accuracy", cv=folds, n_jobs=-1
    )

    pipeline.fit(X_train["x"], y_train)
    y_pred = pipeline.predict(X_test["x"])
    score = accuracy_score(y_test, y_pred)

    print(pipeline.steps[2][0])
    print(f"Mean CV accuracy: {np.mean(CV_scores)}")
    print(f"Computed in {time.time() - start}s")
    print(f"Test accuracy: {score}")
    print(classification_report(y_test, y_pred))

"""
df_en:
SGD - 92%
LogReg - 92%
Ridge - 91%
"""

# %%
"""FR"""

df_fr_ready = prepare_df(df_fr, lang="fr")

# Train/test split to have an accurate score
X_train, X_test, y_train, y_test = train_test_split(
    df_fr_ready.drop("y", axis=1),
    df_fr_ready.loc[:, "y"],
    random_state=31415,
    test_size=0.2,
    shuffle=True,
)

# %%
for i, pipeline in enumerate(pipelines):
    start = time.time()
    CV_scores = cross_val_score(
        pipeline, X_train["x"], y_train, scoring="accuracy", cv=folds, n_jobs=-1
    )

    pipeline.fit(X_train["x"], y_train)
    y_pred = pipeline.predict(X_test["x"])
    score = accuracy_score(y_test, y_pred)

    print(pipeline.steps[2][0])
    print(f"Mean CV accuracy: {np.mean(CV_scores)}")
    print(f"Computed in {time.time() - start}s")
    print(f"Test accuracy: {score}")
    print(classification_report(y_test, y_pred))

"""
df_fr:
SGD - 91%
LogReg - 91%
Ridge - 91%
"""

# %%
# Experiment: try with "df_other" and treat it as english+french+de+it
"""Other"""

df_other_ready = prepare_df(df_other, lang="all")

# Train/test split to have an accurate score
X_train, X_test, y_train, y_test = train_test_split(
    df_other_ready.drop("y", axis=1),
    df_other_ready.loc[:, "y"],
    random_state=31415,
    test_size=0.2,
    shuffle=True,
)

# %%
for i, pipeline in enumerate(pipelines):
    start = time.time()
    CV_scores = cross_val_score(
        pipeline, X_train["x"], y_train, scoring="accuracy", cv=folds, n_jobs=-1
    )

    pipeline.fit(X_train["x"], y_train)
    y_pred = pipeline.predict(X_test["x"])
    score = accuracy_score(y_test, y_pred)

    print(pipeline.steps[2][0])
    print(f"Mean CV accuracy: {np.mean(CV_scores)}")
    print(f"Computed in {time.time() - start}s")
    print(f"Test accuracy: {score}")
    print(classification_report(y_test, y_pred))

"""
df_other:
SGD - 89%
LogReg - 90%
Ridge - 90%
"""
