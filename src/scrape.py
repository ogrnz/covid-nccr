"""
Main scraping script, run this file to complete
the database with latest tweets
"""

from datetime import date
import time
import json

import tweepy

from common.database import Database
from common.app import App
from common.helpers import Helpers
from common.api import Api
from common.classify import Classifier

if __name__ == "__main__":
    today = date.today()

    # Instanciate needed classes
    app = App(debug=False)
    db = Database("tweets_tests.db")
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
                last_ids[name] = 0
                # print("TypeError", name, error)

    # Connect to the API
    api = Api(app.consumer_key, app.consumer_secret, app.access_key, app.access_secret)

    # Retrieve all new tweets per actor
    t1 = time.time()
    total = len(screen_names)

    total_tweets = {}
    scrape_errors = {}
    for i, actor in enumerate(screen_names, start=1):
        print(f"{i}/{total}")
        print(f"Starting to retrieve tweets for {actor}")

        try:
            last_tweet_id = last_ids[actor]
            tweets = api.get_tweets(actor, last_tweet_id)
            total_tweets[actor] = tweets

        except tweepy.TweepError as error:
            scrape_errors[actor] = str(error)
            print("ERROR", actor, error)

        except KeyError as error:
            print("KeyError when retrieving tweets from", actor, error)
            continue

        except KeyboardInterrupt as error:
            pass

        if app.debug:
            break

    elapsed = time.time() - t1
    Helpers.print_timer(elapsed)
    if len(scrape_errors) > 0:
        print("With some errors:", json.dumps(scrape_errors, indent=4))

    # Classify tweets (about covid or not)
    t1 = time.time()
    tweet_entries = []
    for actor in total_tweets:
        tot_tweets_actor = len(total_tweets[actor])

        for i, tweet in enumerate(total_tweets[actor], start=1):
            Helpers.dynamic_text(
                f"Classifiying tweets from {actor}: \
                {i}/{tot_tweets_actor} \r"
            )

            tweet_entry = ()
            for key, val in total_tweets[actor][tweet].items():
                tweet_entry += (val,)

            tmp_tweet = list(tweet_entry)

            # Check if is about covid
            if classifier.classify(tmp_tweet[7]):
                tmp_tweet[1] = 1  # About covid
            else:
                tmp_tweet[1] = 0  # Not about covid

            tweet_entry = tuple(tmp_tweet)
            tweet_entries.append(tweet_entry)

        if app.debug:
            break

    elapsed = time.time() - t1
    Helpers.print_timer(elapsed)

    # Insert tweets into DB
    t1 = time.time()
    print("Inserting new tweets")
    with db:
        inserted = db.insert_many(tweet_entries)
    print(f"{inserted} tweets inserted")

    elapsed = time.time() - t1
    Helpers.print_timer(elapsed)
