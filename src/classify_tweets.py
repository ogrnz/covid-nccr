"""
Classify existing tweets in the database as
being about covid or not
"""

from common.database import Database
from common.classify import Classifier
from common.helpers import Helpers

if __name__ == "__main__":
    db = Database("tweets.db")
    classifier = Classifier()

    with db:
        print("Downloading from db...")
        tweets = db.get_fields(["tweet_id", "covid_theme", "text"])

    print("Classifying tweets")
    tot = len(tweets)
    classified_tweets = []
    COUNT = 0

    for index, tweet in enumerate(tweets, start=1):
        Helpers.dynamic_text(f"{index}/{tot}")

        tmp_tweet = list(tweet)
        txt = tmp_tweet[2]

        if classifier.classify(txt):
            tmp_tweet[1] = 1
            COUNT += 1

        tweet = (tmp_tweet[1], tmp_tweet[0])
        classified_tweets.append(tweet)

    print(f"Updating {COUNT} classified tweets...")

    with db:
        modified = db.update_theme_many(classified_tweets)

    print(f"{modified} tweets updated.")
