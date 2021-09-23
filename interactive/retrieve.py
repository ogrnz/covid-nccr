#%%
import sys, os

sys.path.append(os.path.abspath(os.path.join("..", "src")))

#%%
%load_ext autoreload
%autoreload 2
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
    tws = db.get_all_tweets()
print(len(tws))
df_all = Helpers.df_from_db(tws)

#%%
# get only tweets about covid that are NOT coded
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
# only keep tweets of the 2 waves
start = "31/12/2019"
end = "01/04/2021"

df_sorted = df[
    (df["date"] > pd.to_datetime(start, format="%d/%m/%Y"))
    & (df["date"] < pd.to_datetime(end, format="%d/%m/%Y"))
]
df_sorted  # 48_009 tweets

#%%
# Of those, only keep the ones from @Left_EU and @Sante_Gouv
handles = ["@Left_EU", "@Sante_Gouv"]
df_sorted = df_sorted[
    df_sorted["handle"].isin(handles)
]
df_sorted  # 1_170 tweets

# %%
df_sorted.to_excel("database/xlsx/missing.xlsx", index=False)

# %%
