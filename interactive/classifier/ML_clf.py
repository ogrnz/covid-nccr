# pylint: skip-file

"""
ML classifier file.

After investigation (see ML_clf_lab.py), the selected models are:
1. SGDClassifier
2. LogisticRegression
3. RidgeClassifier

All have about 91% accuracy for the english and french sets and about 90% for the general `other` set.

For the final classifier, the approach is the following:
1. For each set (en, fr, other), classify each tweets with the 3
algorithms trained on the respective set
2. The final choice is the decision of the majority of the 3 models (2/3)
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
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

#%%
app_run = App(debug=False)
db = Database("tweets.db", app=app_run)

#%%
# Import the sets
df_en = pd.read_pickle("interactive/data/db_en.pkl")
df_fr = pd.read_pickle("interactive/data/db_fr.pkl")
df_other = pd.read_pickle("interactive/data/db_other.pkl")

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


#%%
# General pipeline structure
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
    pipeline = make_pipeline(CountVectorizer(), TfidfTransformer(), model)
    pipelines.append(pipeline)
    pipeline.set_params(**optimal_params[pipeline.steps[2][0]])

# %%
dfs = [df_en, df_fr, df_other]
clf_dfs = []

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
    print("Preprocessing..\n")
    # 1: keep only already coded tweets
    df_coded = df[~df["topic"].isnull() | ~df["theme_hardcoded"].isnull()].copy()
    df_uncoded = df[df["topic"].isnull() & df["theme_hardcoded"].isnull()].copy()

    # 2: Set `y` column
    df_coded.loc[:, "y"] = df_coded.progress_apply(covid_classify, axis=1)
    df_uncoded.loc[:, "y"] = np.nan

    # 3: create new col "x" with text or old_text if text is nan
    df_coded.loc[:, "x"] = df_coded.progress_apply(
        lambda r: r["old_text"] if r["text"] is None else r["text"], axis=1
    )
    df_uncoded.loc[:, "x"] = df_uncoded.progress_apply(
        lambda r: r["old_text"] if r["text"] is None else r["text"], axis=1
    )

    # 4: Sanitize text
    df_coded.loc[:, "x"] = df_coded["x"].progress_apply(
        lambda t: sanitize(t, lang=LANG)
    )
    df_uncoded.loc[:, "x"] = df_uncoded["x"].progress_apply(lambda t: sanitize(t, lang=LANG))

    # 5. train/test split to have an accurate score
    X_train, X_test, y_train, y_test = train_test_split(df_coded.drop("y", axis=1), df_coded.loc[:, "y"], random_state=31415, test_size=0.2, shuffle=True)

    # Predict
    print("Predicting..")
    for i, pipeline in enumerate(pipelines):
        stepname = pipeline.steps[2][0]
        print(stepname)
        start = time.time()

        # Fit
        pipeline.fit(X_train.loc[:, "x"], y_train)

        # Predict test
        y_pred = pipeline.predict(X_test.loc[:, "x"])
        score = accuracy_score(y_test, y_pred)

        print(classification_report(y_test, y_pred))

        # Predict "real"
        df_coded.loc[:, stepname] = pipeline.predict(df_coded.loc[:, "x"])
        df_uncoded.loc[:, stepname] = pipeline.predict(df_uncoded.loc[:, "x"])

        print(f"Computed in {round(time.time() - start, 3)}s")

    df_coded.loc[:, "covid_theme"] = np.nan
    df_coded.loc[:, "covid_theme"] = df_coded.apply(get_theme, axis=1)
    df_uncoded.loc[:, "covid_theme"] = np.nan
    df_uncoded.loc[:, "covid_theme"] = df_uncoded.apply(get_theme, axis=1)

    # Append to classified list
    clf_dfs.append(df_coded)
    clf_dfs.append(df_uncoded)

    # Compute false neg/pos
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

df_final = pd.concat(clf_dfs)
df_final["covid_theme"].unique()

#%%
df_final.to_pickle("interactive/data/db_ML.pkl")

#%%
# After ML, correct with some manual changes
df_final = pd.read_pickle("interactive/data/db_ML.pkl")

def classify(row, clf, mode="real"):
    txt = row.loc["old_text"] if row.loc["text"] is None else row.loc["text"]

    topics_cov = [
        "601",
        "601.0",
        "602",
        "602.0",
        "603",
        "603.0",
        "604",
        "604.0",
        "605",
        "605.0",
        "606",
        "606.0",
        "607",
        "607.0",
    ]
    topics_not_cov = ["608", "608.0"]

    if mode == "fake":
        if (clf.classify(txt) \
            or row.loc["topic"] in topics_cov):
            return 1  # About covid
        elif (row.loc["topic"] in topics_not_cov \
            or row.loc["theme_hardcoded"] == "0"):
            return 0 # Not about covid
        else:
            return row.loc["covid_theme"]
    else:
        if clf.classify(txt):
            return 1  # About covid
        else:
            return row.loc["covid_theme"]


#%%
from common.classify import Classifier
%load_ext autoreload
%autoreload 2

classifier = Classifier(keywords_file="covid_final.txt")

df_final["covid_theme"] = df_final.progress_apply(
    lambda row: classify(row, classifier, mode="real"), axis=1
)

#%%
topics_cov = [
    "601",
    "601.0",
    "602",
    "602.0",
    "603",
    "603.0",
    "604",
    "604.0",
    "605",
    "605.0",
    "606",
    "606.0",
    "607",
    "607.0",
]
topics_not_cov = ["608", "608.0"]


# All coded tweets
all_coded = df_final[
    (df_final["topic"].isin(topics_cov + topics_not_cov)) | (df_final["theme_hardcoded"] == "0")
]
all_yes = all_coded[all_coded["topic"].isin(topics_cov)]
all_no = all_coded[
    all_coded["topic"].isin(topics_not_cov) | (all_coded["theme_hardcoded"] == "0")
]

# %%
# Calculate false negative
false_neg_count = sum(all_yes["covid_theme"] == "0")
all_count = len(all_yes)

print("False Negative:")
print(
    f"Out of {all_count} manually coded tweets about covid, {false_neg_count} were classified as not being about covid although they were. The false negative rate is {round(false_neg_count / all_count*100, 1)}%."
)
"""
"Real" false negative: 0%
"Fake" (with manual changes to have a clean database): 0%
"""
#%%
# Calculate false positive

false_pos_count = sum(all_no["covid_theme"] == 1)
all_excluded_count = len(all_no)

print("False Positive:")
print(
    f"Out of {all_excluded_count} manually classified tweets, {false_pos_count} were classified as being about covid although they were not. The false positive rate is {round(false_pos_count / all_excluded_count*100, 1)}%."
)

"""
"Real" false positive: 6.4%
"Fake" (with manual changes to have a clean database): 4.2%
"""
#%%
# Check a few tweets
df_final[df_final["theme_hardcoded"] == "0"]
df_final["covid_theme"].unique()
df_final[df_final["tweet_id"] == "1212681570710151168"] # 0
df_final[df_final["tweet_id"] == "1287812564"] # 1
df_final[df_final["tweet_id"] == "1248928528525115392"] # 1
# %%
# Extract theme and tweet_id

to_update = []

for row in df_final.iterrows():
    to_update.append((row[1].loc["covid_theme"], row[1].loc["tweet_id"]))

print(len(to_update))

#%%
# Upload to the DB

with db:
    count = db.update_theme_many(to_update)
    print(f"{count} tweets updated")

# %%
