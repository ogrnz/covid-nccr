"""
Classify existing tweets in the database as
being about covid or not
"""

import os
import logging

from tqdm import tqdm

from common.app import App
from common.database import Database
from common.classify import Classifier
from common.helpers import Helpers

log = logging.getLogger(os.path.basename(__file__))

if __name__ == "__main__":
    app_run = App(debug=False)

    db = Database("tweets.db", app=app_run)
    classifier = Classifier()

    with db:
        log.info("Downloading from db...")
        tweets = db.get_fields(["tweet_id", "covid_theme", "text"])

    log.info("Classifying tweets")
    tot = len(tweets)
    classified_tweets = []
    COUNT = 0

    for index, tweet in tqdm(enumerate(tweets, start=1)):

        tmp_tweet = list(tweet)
        txt = tmp_tweet[2]

        if not classifier.classify(txt):
            # If tweet is NOT about covid, set it as 0
            tmp_tweet[1] = 0
            COUNT += 1

        tweet = (tmp_tweet[1], tmp_tweet[0])
        classified_tweets.append(tweet)

    log.info(f"Updating {COUNT} classified tweets...")

    with db:
        inserted = db.update_theme_many(classified_tweets)

    log.info(f"{inserted} Tweets updated.")
