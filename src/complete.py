"""
Complete tweets based on xlsx file
"""


import time

import pandas as pd

from common.app import App
from common.api import Api
from common.helpers import Helpers


def complete(filename):
    """
    Main script
    """

    (ids_xls, xls_df) = Helpers.extract_ids_file(
        f"src/resources/data/{filename}.xlsx", retdf=True
    )

    print(f"Starting length: {len(ids_xls)}")
    completed_tweets = api.get_tweets_by_ids_with_nan(ids_xls, xls_df)

    print(f"Finishing length: {len(completed_tweets)}")

    print(completed_tweets)
    # completed_tweets.to_pickle(f"src/resources/data/{filename}.pkl")
    completed_tweets.to_excel(f"src/resources/data/{filename}_total.xlsx")


if __name__ == "__main__":
    app = App(debug=False)
    api = Api(
        app.consumer_key, app.consumer_secret, app.access_key, app.access_secret, app
    )

    t1 = time.time()

    # Input files to complete here
    # missing values with full dataset
    files = ["full1"]
    for xls in files:
        complete(xls)

    elapsed = time.time() - t1
    Helpers.print_timer(elapsed)
