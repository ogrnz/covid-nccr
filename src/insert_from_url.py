"""
Update database from list of URLs
Code is similar to scrape.py
"""

import pandas as pd

from common.app import App
from common.database import Database
from common.api import Api
from common.helpers import Helpers


if __name__ == "__main__":
    app_run = App(debug=True)
    db = Database("tweets.db", app=app_run)
    filename = "missing.xlsx"

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

    completed_tweets = pd.read_pickle(f"src/resources/data/{filename}.pkl")

    # Insert tweets into db
    tweet_entries = [tuple(entry) for entry in completed_tweets.to_numpy()]
    with db:
        inserted = db.insert_many(tweet_entries)

    print(f"Done inserting {inserted} tweets")

    # Remember to classify the database if you need to
