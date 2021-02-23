"""
Main scraping script, run this file to complete
the database with latest tweets
"""

from datetime import date
import time
import json

from common.database import Database
from common.app import App
from common.helpers import Helpers
from common.api import Api

today = date.today()

app = App(debug=False)
db = Database("tweets_tests.db")

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
            print("TypeError", name, error)

# Connect to the API
api = Api(app.CONSUMER_KEY, app.CONSUMER_SECRET, app.ACCESS_KEY, app.ACCESS_SECRET)

t1 = time.time()
total = len(screen_names)

total_tweets = {}
scrape_errors = {}

# Retrieve all new tweets per actor
for i, actor in enumerate(screen_names, start=1):
    print(f"{i}/{total}")
    print(f"Starting to retrieve tweets for {actor}")

    last_tweet_id = last_ids[actor]
    try:
        tweets = api.get_tweets(actor, last_tweet_id)
        total_tweets[actor] = tweets
    except Exception as error:
        scrape_errors[actor] = str(error)
        print("ERROR", actor, error)

    if app.DEBUG:
        break

elapsed = time.time() - t1
print(f"Done in {round(elapsed / 60, 2)} min")
if len(scrape_errors) > 0:
    print("With some errors:", json.dumps(scrape_errors, indent=4))

# Insert tweets into DB
t1 = time.time()
db_errors = {}

for actor in total_tweets:
    print("Inserting tweets from", actor)
    tot_tweets_actor = len(total_tweets[actor])

    with db:
        for i, tweet in enumerate(total_tweets[actor], start=1):
            Helpers.dynamic_text(f"{i}/{tot_tweets_actor}")

            tweet_entry = ()
            for key, val in total_tweets[actor][tweet].items():
                tweet_entry += (val,)

            tmp_tweet = list(tweet_entry)
            # Work with entries
            # Check if is about covid
            tmp_tweet[1] = 0
            tweet_entry = tuple(tmp_tweet)

            try:
                db.insert_tweet(tweet_entry)
            except Exception as error:
                db_errors[actor][tweet] = str(error)
                print("ERROR", actor, error)

elapsed = time.time() - t1
print(f"Done in {round(elapsed / 60, 2)} min")
if len(db_errors) > 0:
    print("With some errors:", json.dumps(db_errors, indent=4))
