# pylint: skip-file

"""
Detect language (en, fr, other) of a tweet.
Currently used in the classifier.
"""

#%%
import sys, os

sys.path.append(os.path.abspath(os.path.join("..", "..", "src")))

#%%
import math

import pandas as pd
import numpy as np

import langid
from langid.langid import LanguageIdentifier, model

identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

from tqdm import tqdm
import tweepy

from common.database import Database
from common.app import App
from common.helpers import Helpers
from common.api import Api

#%%
app_run = App(debug=False)
db = Database("tweets.db", app=app_run)
api = Api(main_app=app_run)
data_path = os.path.join(app_run.root_dir, "interactive", "data")

#%%
with db:
    tweets = db.get_all_tweets()
len(tweets)

# %%
df = pd.DataFrame(
    tweets,
    columns=Helpers.schema_cols,
)
df["url"].isna().sum()
# 267 NA's -> tweets with no URL, only '@MinSoliSante' tweets (account deleted)

#%%
identifier.classify(df["text"][0])
# works!

#%%
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


# %%
# If `text` is empty, use `old_text`
subs = df[df["theme_hardcoded"] == "0"].head(100)
subs["lang"] = subs.apply(
    lambda row: lang_detect(row["old_text"])
    if row["text"] is None
    else lang_detect(row["text"]),
    axis=1,
)

subs["lang"].unique()
# "en", "fr", "de", "it", nan, "am"

# %%
# For whole dataset
tqdm.pandas()
df["lang"] = df.progress_apply(
    lambda row: lang_detect(row["old_text"])
    if row["text"] is None
    else lang_detect(row["text"]),
    axis=1,
)
df["lang"].unique()

#%%
df.to_pickle(os.path.join(data_path, "db_all_lang.pkl"))

#%%
# If lang is not en or fr, set it to `other`
lang_lst = ["en", "fr"]
df["lang"] = df["lang"].progress_apply(lambda x: "other" if x not in lang_lst else x)
df["lang"].unique()

"""
en - 138958
fr - 82196
other - 19221
"""

# %%
df.to_pickle(os.path.join(data_path, "db_sub_lang.pkl"))

# %%
# Read
df_all = pd.read_pickle(os.path.join(data_path, "db_sub_lang.pkl"))
df_en = df_all[df_all["lang"] == "en"]
df_fr = df_all[df_all["lang"] == "fr"]
df_other = df_all[df_all["lang"] == "other"]

# %%
# Write to pickle
df_en.to_pickle(os.path.join(data_path, "db_en.pkl"))
df_fr.to_pickle(os.path.join(data_path, "db_fr.pkl"))
df_other.to_pickle(os.path.join(data_path, "db_other.pkl"))

# %%
