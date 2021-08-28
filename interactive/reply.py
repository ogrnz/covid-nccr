# pylint: skip-file
"""
Add "RY @handle: ..." to all replies
"""

#%%
import sys, os

sys.path.append(os.path.abspath(os.path.join("..", "src")))

#%%
import pandas as pd
import numpy as np

from common.database import Database
from common.app import App
from common.api import Api

app_run = App(debug=True)
db = Database("tweets.db", app=app_run)

# %%
# Connect to the API
api = Api(
    app_run.consumer_key,
    app_run.consumer_secret,
    app_run.access_key,
    app_run.access_secret,
    main_app=app_run,
)

# %%
with db:
    tweets = db.get_by_type("Reply")
len(tweets)

#%%
df_replies = pd.DataFrame(
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
df_replies["oldText"].fillna("text", inplace=True)

df = df_replies[~df_replies["oldText"].str.startswith("RY")]
print(len(df))

#%%
tweets_ids = df["tweet_id"].values.tolist()
print(len(tweets_ids))

#%%
tot_tweets = None
tot_tweets = api.get_tweets_by_ids_with_nan(tweets_ids, df, no_id_remove=True)

#%%
print(len(tot_tweets))

to_update = [
    (
        tweet[5],
        tweets_ids[i],
    )
    for i, tweet in enumerate(tot_tweets.values.tolist())
]

# %%
with db:
    count = db.update_many("oldText", "tweet_id", to_update)

print(count, "tweets updated")

# %%
# Re-download tweets and check issues
with db:
    tweets = db.get_by_type("Reply")

print(len(tweets))
tweets_df = pd.DataFrame(
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
totprobs = len(tweets_df) - sum(
    tweets_df["oldText"].apply(lambda x: str(x).startswith("RY"))
)
print(totprobs)
# = 26 tweets with an issue. Most of them answers to deleted tweets

# %%
