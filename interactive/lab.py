# pylint: skip-file

#%%
import sys, os

sys.path.append(os.path.abspath(os.path.join("..", "src")))

#%%
import pandas as pd
import numpy as np

from common.database import Database
from common.app import App
from common.api import Api
from common.helpers import Helpers

app_run = App(debug=True)
db = Database("tweets.db", app=app_run)

# %%
# Connect to the API
api = Api(app_run)

# %%
tw = api.get_tweets_by_ids([1215573600821903360])
tw

# %%
def update_df(df: pd.DataFrame, d: dict):
    df.loc[df["tweet_id"] == d["tweet_id"], d.keys()] = d.values()


# %%
with db:
    tweets = db.get_all_tweets()

# %%
