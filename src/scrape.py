"""
Main scraping script, run this file to complete
the database with latest tweets
"""

import os
import time
import json
import logging

import tweepy
from tqdm import tqdm

from common.app import App
from common.api import Api
from common.database import Database
from common.helpers import Helpers
from common.classify import Classifier

log = logging.getLogger(os.path.basename(__file__))


def main(app: App, db: Database):
    # Instanciate needed class
    classifier = Classifier()

    # Retrieve urls and screen_names
    urls = Helpers.get_actors_url("actors_url.txt")
    screen_names = Helpers.get_screen_names(urls)

    # Get last ids per actor from database
    last_ids = {}
    with db:
        for name in screen_names:
            try:
                last_ids[name] = int(db.get_last_id_by_handle(name)[0])
            except TypeError as error:
                last_ids[name] = 0  # Retrieve all tweets

    # Connect to the API
    api = Api()

    # Retrieve all new tweets per actor
    t1 = time.perf_counter()
    total = len(screen_names)

    total_tweets = {}
    scrape_errors = {}

    for i, actor in tqdm(enumerate(screen_names, start=1), total=total):
        log.debug(f"Starting to retrieve tweets for {actor}")

        try:
            last_tweet_id = last_ids[actor]
            tweets = api.get_tweets(actor, last_tweet_id)
            total_tweets[actor] = tweets

        except tweepy.TweepError as error:
            scrape_errors[actor] = str(error)
            log.warning("Error", actor, error)

        except KeyError as error:
            log.warning("KeyError when retrieving tweets from", actor, error)
            continue

        except KeyboardInterrupt as error:
            log.info("KeyboardInterrupt: stopping script")
            exit()

        if app.debug:
            log.debug("Breaking loop")
            break

    elapsed = time.time() - t1
    Helpers.print_timer(elapsed)
    if len(scrape_errors) > 0:
        log.info("With some errors:", json.dumps(scrape_errors, indent=4))

    # Classify tweets (about covid or not)
    t1 = time.time()
    tweet_entries = []
    for actor in total_tweets:
        tot_tweets_actor = len(total_tweets[actor])
        log.info(f"Classifiying tweets from {actor}")

        for _, tweet in enumerate(total_tweets[actor], start=1):
            tweet_entry = ()
            for _, val in total_tweets[actor][tweet].items():
                tweet_entry += (val,)

            tmp_tweet = list(tweet_entry)

            # Check if is about covid
            if not classifier.classify(tmp_tweet[7]):
                tmp_tweet[1] = 0  # Not about covid
            else:
                tmp_tweet[1] = 1  # About covid

            tweet_entry = tuple(tmp_tweet)
            tweet_entries.append(tweet_entry)

        if app.debug:
            break

    elapsed = time.perf_counter() - t1
    Helpers.print_timer(elapsed)

    # Insert tweets into DB
    t1 = time.perf_counter()
    log.info("Inserting new tweets")
    with db:
        inserted = db.insert_many(tweet_entries)
    log.info(f"{inserted} tweets inserted")

    elapsed = time.perf_counter() - t1
    Helpers.print_timer(elapsed)


if __name__ == "__main__":
    app_run = App()
    database = Database("tweets_test.db", app=app_run)

    main(app_run, database)
