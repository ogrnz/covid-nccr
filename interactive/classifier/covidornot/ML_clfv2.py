# pylint: skip-file

"""
ML classifier file.

After investigation (see ML_clf_lab.py), the selected models are:
1. SGDClassifier
2. LogisticRegression
3. RidgeClassifier

All have about 94% accuracy for the english and french sets and about 92% for the general `other` set.

For the final classifier, the approach is the following:
1. For each set (en, fr, other), classify each tweets with the 3
algorithms trained on the respective set
2. The final choice is the decision of the majority of the 3 models (2/3)
"""

#%%
import sys, os

sys.path.append(os.path.abspath(os.path.join("src")))

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

tqdm.pandas()

from common.app import App
from common.database import Database
from common.helpers import Helpers

from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#%%
app_run = App(debug=False)
db = Database("tweets.db", app=app_run)
data_path = os.path.join(app_run.root_dir, "interactive", "data")

#%%
# Import the sets
df_en = pd.read_pickle(os.path.join(data_path, "db_en.pkl"))
df_fr = pd.read_pickle(os.path.join(data_path, "db_fr.pkl"))
df_other = pd.read_pickle(os.path.join(data_path, "db_other.pkl"))

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
    pipeline = make_pipeline(CountVectorizer(preprocessor=None), TfidfTransformer(), model)
    pipelines.append(pipeline)
    pipeline.set_params(**optimal_params[pipeline.steps[2][0]])

# %%
dfs = [df_en, df_fr, df_other]
# dfs = [df_other]
clf_dfs = []
error_rates = {}

for j, df in enumerate(dfs):
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

    # 3: update col "text" with text or old_text if text is nan
    df_coded.loc[:, "text"] = df_coded.progress_apply(
        lambda r: r["old_text"] if r["text"] is None else r["text"], axis=1
    )
    df_uncoded.loc[:, "text"] = df_uncoded.progress_apply(
        lambda r: r["old_text"] if r["text"] is None else r["text"], axis=1
    )

    # 4: Sanitize text and set `x` col
    df_coded.loc[:, "x_text"] = df_coded["text"].progress_apply(
        lambda t: sanitize(t, lang=LANG)
    )
    df_uncoded.loc[:, "x_text"] = df_uncoded["text"].progress_apply(lambda t: sanitize(t, lang=LANG))

    df_coded.loc[:, "x_handle"] = df_coded["handle"]
    df_uncoded.loc[:, "x_handle"] = df_uncoded["handle"]

    df_coded.loc[:, "x_date"] = df_coded["created_at"].progress_apply(
        lambda r: r[:10].replace("-", "/"),
    )
    df_uncoded.loc[:, "x_date"] = df_uncoded["created_at"].progress_apply(
        lambda r: r[:10].replace("-", "/"),
    )

    df_coded.loc[:, "x"] = df_coded.progress_apply(
        lambda r: f'{r["x_date"]} {r["x_handle"]} {r["x_text"]}', axis=1
    )
    df_uncoded.loc[:, "x"] = df_uncoded.progress_apply(
        lambda r: f'{r["x_date"]} {r["x_handle"]} {r["x_text"]}', axis=1
    )

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
    print(f"\nFalse negative: {round(false_neg * 100, 2)}")

    df_yes = df_coded[
        ~(df_coded["theme_hardcoded"] == "0")
        & ~(df_coded["topic"] == "608")
        & ~(df_coded["topic"] == "608.0")
    ].copy()
    false_pos = 1 - (df_yes["covid_theme"] == 1).sum() / len(df_yes)
    print(f"False positive: {round(false_pos * 100, 2)}")

    error_rates[j] = {"False negative rate (%)": round(false_neg * 100, 2), "False positive rate (%)": round(false_pos * 100, 2)}

df_final = pd.concat(clf_dfs)
df_final["covid_theme"].unique()  # small verification
print(error_rates)

#%%
# Save pipelines to pkl to reuse
pickle.dump(pipelines, open(os.path.join(data_path, "models", "sgd_logreg_ridge_23092021.pkl"), "wb"))

#%%
# Save db after only ML classification
df_final.to_pickle(os.path.join(data_path, "db_ML.pkl"))

#%%
# After ML, correct with some manual changes
df_final = pd.read_pickle(os.path.join(data_path, "db_ML.pkl"))

# Keep the columns in the pkl file for debugging purposes
df_final = df_final.drop(columns=["y", "x_text", "x_handle", "x_date", "x", "sgdclassifier", "logisticregression", "ridgeclassifier"])

#%%
def classify(row, clf, mode="real"):
    """
    Postprocessing classifying function.
    Allows to add post-treatment rules.

    In the "real" mode, we check each tweet against the naive classifier again, to be sure that if a tweet containing obvious covid keywords ("covid", "ncov", ...), it will be classifier as being about covid.

    The "fake" mode is here to fix inconsistencies in the database, but artificially betters the false pos/neg rates. For example if an already-coded tweet has a topic other than 608, it cannot be classified as not being about covid, so we override the ML classifier decision.
    """

    txt = row.loc["old_text"] if row.loc["text"] is None else row.loc["text"]

    if mode == "fake":
        if (clf.classify(txt)
            or row.loc["topic"] in Helpers.topics_cov):
            return 1  # About covid
        elif (row.loc["topic"] in Helpers.topics_not_cov
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

#%%
df_final["covid_theme"] = df_final.progress_apply(
    lambda r: classify(r, classifier, mode="fake"), axis=1
)

#%%
# Save final db after ML classification + keyword classification
df_final.to_pickle(os.path.join(data_path, "db_final_classified.pkl"))

#%%
# All coded tweets

all_coded = df_final[
    (df_final["topic"].isin(Helpers.topics_cov + Helpers.topics_not_cov)) | (df_final["theme_hardcoded"] == "0")
]
all_yes = all_coded[all_coded["topic"].isin(Helpers.topics_cov)]
all_no = all_coded[
    all_coded["topic"].isin(Helpers.topics_not_cov) | (all_coded["theme_hardcoded"] == "0")
]

# %%
# Calculate false negative
false_neg_count = sum(all_yes["covid_theme"] == "0")
all_count = len(all_yes)

print("False Negative:")
print(
    f"Out of {all_count} manually coded tweets about covid, {false_neg_count} were classified as not being about covid although they were. The false negative rate is {round(false_neg_count / all_count*100, 2)}%."
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
    f"Out of {all_excluded_count} manually classified tweets, {false_pos_count} were classified as being about covid although they were not. The false positive rate is {round(false_pos_count / all_excluded_count*100, 2)}%."
)

"""
"Real" false positive: 2.4%
"Fake" (with manual changes to have a clean database): 0.12%
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

for row in tqdm(df_final.iterrows()):
    to_update.append((row[1].loc["covid_theme"], row[1].loc["tweet_id"]))

print(len(to_update))

#%%
# Upload to the DB

with db:
    count = db.update_theme_many(to_update)
    print(f"{count} tweets updated")

# %%
