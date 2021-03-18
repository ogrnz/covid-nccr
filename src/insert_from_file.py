"""
Update database from xls file
Code is similar to scrape.py

WARNING: The columns of the file must absolutely
match the sql schema EXCEPT for the first "tweet_id" and "covid_theme" cols:
columns=[
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
]
"""

#%%
import uuid
import hashlib

import numpy as np
import pandas as pd

from common.app import App
from common.database import Database
from common.api import Api
from common.helpers import Helpers

#%%
app_run = App(debug=True)
db = Database("test3.db", app=app_run)

cols_schema = 13

# filename = "UN_Mobility.xlsx"
filename = "UN.xlsx"

xls = pd.read_excel(f"src/resources/data/{filename}")
# xls.to_pickle(f"src/resources/data/{filename}.pkl")
# xls = pd.read_pickle(f"src/resources/data/{filename}.pkl")

if xls.shape[1] > cols_schema:
    print("Too many cols", xls.shape)
    xls.drop(xls.columns[12:].tolist(), axis=1, inplace=True)
    print("Final shape:", xls.shape)

# Add tweet_id and covid_theme columns
xls["covid_theme"] = 1
xls["tweet_id"] = None


# Extract ids
xls["tweet_id"] = xls["URL"].apply(Helpers.extract_id)

# If tweet_id==0, then it's na
# hash the tweet with the date, oldText and text
# and use it as id
mask = xls["tweet_id"] == 0

xls.loc[mask, ["tweet_id"]] = (
    xls[mask]["created_at"].astype(str)
    + xls[mask]["oldText"].astype(str)
    + xls[mask]["text"].astype(str)
)
xls.loc[mask, ["tweet_id"]] = xls["tweet_id"].apply(
    lambda x: int(str(int(hashlib.sha1(bytes(x, "utf-8")).hexdigest(), 16))[:10])
)

# Reorder columns
cols = xls.columns.tolist()
cols.insert(0, "covid_theme")
cols.insert(0, "tweet_id")
del cols[-2:]
xls = xls[cols]

# Insert tweets into db
tweet_entries = [tuple(entry) for entry in xls.to_numpy()]

with db:
    inserted = db.insert_many(tweet_entries)

print(f"Done inserting {inserted} tweets")

# Remember to classify the database if you need to
