"""
Update database from list of URLs
Code is similar to scrape.py
"""

# import numpy as np
# import pandas as pd

from common.app import App
from common.database import Database
from common.api import Api
from common.helpers import Helpers


if __name__ == "__main__":
    app_run = App(debug=True)
    database = Database("tweets_tests.db", app=app_run)
    filename = "missing.xlsx"

    # Get URLs
    ids_xls = Helpers.extract_ids_file(f"src/resources/data/{filename}")
    print(len(ids_xls))

    # Connect to the API
    api = Api(
        app_run.consumer_key,
        app_run.consumer_secret,
        app_run.access_key,
        app_run.access_secret,
        main_app=app_run,
    )

    # Get complete tweet information
    completed_tweets = api.get_complete_tweets_by_ids(ids_xls)

    print(completed_tweets[:2])
    print(type(completed_tweets))
    print(len(completed_tweets))
