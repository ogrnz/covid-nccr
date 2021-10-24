"""
Update database from xls file
Code is similar to scrape.py

WARNING: The columns of the file must absolutely
match the sql schema EXCEPT for the first "tweet_id" and "covid_theme" cols:
columns_xls=[
    "created_at",
    "handle",
    "name",
    "old_text",
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

A "theme_hardcoded" is created by default with None values if non-existent
"""

import hashlib

import pandas as pd
from tqdm import tqdm

from common.app import App
from common.database import Database
from common.helpers import Helpers

app_run = App(debug=True)
db = Database("tweets.db", app=app_run)


files = ["excluded.xlsx"]

for filename in tqdm(files):
    xls = pd.read_excel(f"src/resources/data/{filename}")

    # Remove empty columns
    xls = xls.loc[:, ~xls.columns.str.startswith("Unnamed:")]

    # Add theme_hardcoded if it does not exist
    if "theme_hardcoded" not in xls.columns:
        xls["theme_hardcoded"] = None

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
        lambda x: str(int(hashlib.sha1(bytes(x, "utf-8")).hexdigest(), 16))[:10]
    )

    # Reorder columns
    cols = xls.columns.tolist()
    cols.remove("covid_theme")
    cols.remove("tweet_id")
    cols.insert(0, "covid_theme")
    cols.insert(0, "tweet_id")
    xls = xls[cols]

    # Insert tweets into db
    tweet_entries = [tuple(entry) for entry in xls.to_numpy()]

    # with db:
    #     inserted = db.insert_or_replace_many(tweet_entries)

    # print(f"Done inserting {inserted} tweets")

    # Remember to classify the database if needed

    print(tweet_entries)
# TODO
# Sanitize strings (`type`) before inserting
# trail spaces and such...
