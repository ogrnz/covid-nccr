# pylint: skip-file

"""
Download and keep only the tweets that we are interested in from the db.
-> Tweets in the range (first 2 waves) that are [...]
"""

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
df_all = Helpers.df_from_db(tweets)

#%%
# and sort it
df_yes = df_all[(df_all["covid_theme"] == 1) & ~(df_all["theme_hardcoded"] == "0")]
df = df_yes[
    ~(df_yes["topic"].isin(Helpers.topics_cov))
    & ~(df_yes["topic"].isin(Helpers.topics_not_cov))
].copy()

#%%
def convert_date(datestr):
    datestr = datestr.split(" ")[0]
    return datestr.replace("-", "/")


#%%
# Convert date to be handled
df["date"] = df["created_at"].apply(convert_date)
df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")


# %%
# only keep between 2020/09/01 and 2021/03/31
start = "31/08/2020"
end = "01/04/2021"

df_sorted = df[
    (df["date"] > pd.to_datetime(start, format="%d/%m/%Y"))
    & (df["date"] < pd.to_datetime(end, format="%d/%m/%Y"))
]
df_sorted
# 28931 tweets

# %%
df_sorted.to_excel("database/xlsx/Full_2nd_wave.xlsx", index=False)

# %%
