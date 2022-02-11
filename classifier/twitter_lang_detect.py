# pylint: skip-file

# %%
"""
Other experiment, non conclusive
Check if we can get the lang from twitter for the `other` category.

Conclusion:
I compared the results with the `langid` module and the `lang` attribute from the twitter API for the `df_other` tweets (language was set with a confidence < 0.9).

From those, only 2552 tweets where english (either from twitter or me) and around 450 in french, so not worth the effort.
"""

import sys, os

sys.path.append(os.path.abspath(os.path.join("..", "..", "src")))

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

df_other = pd.read_pickle("interactive/data/db_other.pkl")

# %%
tw_id = [1260578819590799366]
tweet = api.api.statuses_lookup(id_=tw_id, tweet_mode="extended")
tweet[0].lang

#%%
def get_tweets_by_ids_with_nan(df):
    # Otherwise too long to handle
    df["tweet_id"].astype(str)
    tweets_ids = df["tweet_id"].values.tolist()
    lim_start = 100
    start = 0
    final = lim_start + Helpers.count_null_id(tweets_ids, finish=lim_start)
    ids = tweets_ids[start:final]

    iter_needed = math.ceil((len(tweets_ids) - Helpers.count_null_id(tweets_ids)) / 100)

    print("Completing tweets..")
    itera = 0

    while itera < iter_needed + 1:
        Helpers.dynamic_text(f"{itera}/{iter_needed}")

        print(f"{itera}/{iter_needed}")

        try:
            compl_tweets = api.api.statuses_lookup(id_=ids, tweet_mode="extended")
        except tweepy.TweepError as error:
            if error.api_code == 38:
                # End of loop
                break
            print("Error", error)

        # Edit tot_tweets dict
        for compl_tweet in compl_tweets:
            compl_tweet.id = str(compl_tweet.id)
            df.loc[df["tweet_id"] == compl_tweet.id, ["lang"]] = compl_tweet.lang

        # Setup request for new iteration
        last_iter_final = final
        start = last_iter_final
        if start + 100 <= len(tweets_ids):
            final = (
                last_iter_final
                + 100
                + Helpers.count_null_id(tweets_ids, start=start, finish=start + 100)
            )
        else:
            final = len(tweets_ids)

        ids = tweets_ids[start:final]
        itera += 1

    # Reorder cols
    cols = df.columns.tolist()
    df = df[cols]
    # print(cols)
    return df


#%%
df_other_new = get_tweets_by_ids_with_nan(df_other)

# %%
df_other["lang_tw"] = df_other_new["lang"]
# %%
(
    df_other[df_other["lang_tw"] == df_other["lang"]]["lang"] == "en"
).sum()  # exactly the same

#%%
tqdm.pandas()
df_other["lang"] = df_other.progress_apply(
    lambda row: lang_detect(row["old_text"], threshold=0.8)
    if row["text"] is None
    else lang_detect(row["text"], threshold=0.8),
    axis=1,
)
df_other["lang"].unique()
