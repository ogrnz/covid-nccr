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

import pandas as pd

from common.app import App
from common.database import Database
from common.api import Api
from common.helpers import Helpers


if __name__ == "__main__":
    app_run = App(debug=True)
    db = Database("tweets_tests.db", app=app_run)
    filename = "FR.xlsx"

    # xls = pd.read_excel(f"src/resources/data/{filename}")
    # xls.to_pickle(f"src/resources/data/{filename}.pkl")
    xls = pd.read_pickle(f"src/resources/data/{filename}.pkl")
    print(xls.head())

    # Add tweet_id and covid_theme columns
    xls["covid_theme"] = 1
    print(xls.head())

    # Get URLs
    # ids_xls = Helpers.extract_ids_file(f"src/resources/data/{filename}")
    # print(len(ids_xls))

    # Connect to the API
    # api = Api(
    #     app_run.consumer_key,
    #     app_run.consumer_secret,
    #     app_run.access_key,
    #     app_run.access_secret,
    #     main_app=app_run,
    # )

    # Get complete tweet information
    # completed_tweets = api.get_tweets_by_ids(ids_xls)
    # completed_tweets.to_pickle(f"src/resources/data/{filename}.pkl")

    # Insert tweets into db
    # tweet_entries = [tuple(entry) for entry in completed_tweets.to_numpy()]
    # with db:
    #    inserted = db.insert_many(tweet_entries)

    # print(f"Done inserting {inserted} tweets")

    # Remember to classify the database if you need to
