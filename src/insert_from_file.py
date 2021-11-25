"""
Update database from xls file
Code is similar to scrape.py

WARNING: The columns of the file must absolutely
match the sql schema EXCEPT for the first "covid_theme" col.
"tweet_id" is dynamically created if not existing.

columns_xls=[
    "tweet_id",
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


def main():
    files = ["prepare_insert.xlsx"]

    for filename in files:
        print("Inserting new tweets from ", filename)

        xls = pd.read_excel(f"src/resources/data/{filename}")
        xls_size = len(xls)

        # Remove empty columns
        xls = xls.loc[:, ~xls.columns.str.startswith("Unnamed:")]

        # Add theme_hardcoded if it does not exist
        if "theme_hardcoded" not in xls.columns:
            xls["theme_hardcoded"] = None

        # Add covid_theme column
        xls["covid_theme"] = 1

        # Check if tweet_id already exists
        # If not create the column and extract ids
        if "tweet_id" not in xls.columns.tolist():
            xls["tweet_id"] = None
            xls["tweet_id"] = xls["URL"].apply(Helpers.extract_id)

        # If tweet_id==0, then it's na
        # hash the tweet with the date, old_text and text
        # and use it as id
        mask = xls["tweet_id"] == 0
        xls.loc[mask, ["tweet_id"]] = (
            xls[mask]["created_at"].astype(str)
            + xls[mask]["old_text"].astype(str)
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

        with db:
            db_size_before = db.get_db_size()
            inserted = db.insert_or_replace_many(tweet_entries)
            db_size_after = db.get_db_size()

        print(f"Done inserting {inserted} tweets")

        # Sanity check
        print(f"{xls_size} tweets to insert")
        print(f"{db_size_before} database size before insertion")
        print(f"{db_size_after} database size before insertion")

        # Remember to classify the database if needed


if __name__ == "__main__":
    main()

# TODO
# Sanitize strings (`type`) before inserting
# trail spaces and such...
