# pylint: skip-file
#%%
import sys, os

sys.path.append(os.path.abspath(os.path.join("..", "src")))

import re
import time

import pandas as pd
import numpy as np
from tqdm import tqdm

tqdm.pandas()

from common.database import Database
from common.app import App
from common.helpers import Helpers

app_run = App(debug=True)
db = Database("tweets.db", app=app_run)

# %%
with db:
    tweets = db.get_all_tweets()
len(tweets)

# %%
df = Helpers.df_from_db(tweets)

# and sort it
df = df[(df["covid_theme"] == 1) & ~(df["theme_hardcoded"] == "0")]
df = df[
    ~(df["topic"].isin(Helpers.topics_cov))
    & ~(df["topic"].isin(Helpers.topics_not_cov))
]

#%%
def convert_date(datestr):
    return datestr.split(" ")[0]


#%%
# Convert date to be handled
df["created_at"] = df["created_at"].apply(convert_date)
df["created_at"] = pd.to_datetime(df["created_at"])


# %%
# only keep between 2020/09/01 and 2021/03/31
df = df[
    (df["created_at"] >= pd.Timestamp(2020, 9, 1))
    & (df["created_at"] <= pd.Timestamp(2021, 3, 31))
]
df
# %%
