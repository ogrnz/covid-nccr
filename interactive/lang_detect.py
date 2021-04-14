# pylint: skip-file

"""
Detect language (en, fr, de)
"""

#%%
import sys, os

sys.path.append(os.path.abspath(os.path.join("..", "src")))

#%%
import pandas as pd
import numpy as np

import langid
from tqdm import tqdm

from common.database import Database
from common.app import App
from common.helpers import Helpers
from common.api import Api

#%%
app_run = App(debug=False)
db = Database("tweets.db", app=app_run)

#%%
with db:
    tweets = db.get_all_tweets()
len(tweets)

# %%
df = pd.DataFrame(
    tweets,
    columns=[
        "tweet_id",
        "covid_theme",
        "created_at",
        "handle",
        "name",
        "oldText",
        "text",
        "URL",
        "type",
        "retweets",
        "favorites",
        "topic",
        "subcat",
        "position",
        "frame",
        "theme_hardcoded",
    ],
)
df["text"].isna().sum()
# 3902 NA's -> tweets with no URL

# %%
from langid.langid import LanguageIdentifier, model

identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

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
subs = df.head(100)
subs["lang"] = subs["text"].apply(lang_detect)

subs["lang"].unique()
# "en", "fr", "de", "it", nan, "am"

# %%
# For whole dataset
tqdm.pandas()
df["lang"] = df["text"].progress_apply(lang_detect)
df["lang"].unique()

#%%
df.to_pickle("interactive/data/db_all_lang.pkl")
#%%
# If lang is not en, fr or de, set it to None
lang_lst = ["en", "fr"]
df["lang"] = df["lang"].apply(lambda x: "other" if x not in lang_lst else x)
df["lang"].unique()

"""
en - 95425
fr - 58090
other - 17291
"""

# %%
df.to_pickle("interactive/data/db_sub_lang.pkl")

# %%
