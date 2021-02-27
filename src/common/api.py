"""
API module
"""

import math
from datetime import datetime as dt

import tweepy
import pandas as pd

from common.app import App
from common.helpers import Helpers


class Api:
    """
    Handles communication with the Twitter API through
    tweepy
    """

    api = None
    connected = False

    def __init__(self, cons_key, cons_scrt, acc_key, acc_scrt, main_app: App = None):
        self.app = main_app
        if App is None:
            self.app = App()
        self.connect_api(cons_key, cons_scrt, acc_key, acc_scrt)

    def connect_api(self, cons_key, cons_scrt, acc_key, acc_scrt):
        """
        Connect to twitter API through tweepy
        """

        try:
            auth = tweepy.OAuthHandler(cons_key, cons_scrt)
            auth.set_access_token(acc_key, acc_scrt)
            self.api = tweepy.API(auth)
            self.connected = True
        except tweepy.TweepError as error:
            print("Error connecting to Twitter API ", error)

    def get_tweets(self, screen_name, last_id=0):
        """
        Retrieve tweets from a user
        Script from @yanofsky as baseline
        https://gist.github.com/yanofsky/5436496
        """

        all_tweets = []
        new_tweets = self.api.user_timeline(
            screen_name=screen_name, count=200, tweet_mode="extended"
        )
        all_tweets.extend(new_tweets)
        oldest = all_tweets[-1].id - 1

        while len(new_tweets) > 0:
            # If tweet older than that ID (== 31/12/2019)
            # or older than last ID in db for that actor, go to next actor
            if oldest < 1211913001147740161 or oldest < last_id:
                break

            print(f"Getting tweets before {oldest} ({screen_name})")

            new_tweets = self.api.user_timeline(
                screen_name=screen_name, count=200, max_id=oldest, tweet_mode="extended"
            )
            all_tweets.extend(new_tweets)
            oldest = all_tweets[-1].id - 1

            print(f"...{len(all_tweets)} tweets downloaded")

        outtweets = {}
        for index, tweet in enumerate(all_tweets):
            # Ignore tweets older than 2019/12/31
            as_of = dt.strptime("2019/12/31", "%Y/%m/%d")
            if tweet.created_at < as_of:
                continue

            old_text = None
            if hasattr(tweet, "retweeted_status"):  # Is RT
                old_text = tweet.full_text
                tweet_type = "Retweet"
                full_text = tweet.retweeted_status.full_text
            else:  # Not a Retweet
                full_text = tweet.full_text
                tweet_type = "New"
                if tweet.in_reply_to_status_id is not None:
                    tweet_type = "Reply"

            # Make final dict
            # Warning: structure should match database's schema!
            outtweets[index] = {
                "tweet_id": tweet.id,
                "covid_theme": None,
                "created_at": tweet.created_at.strftime("%d/%m/%Y %H:%M:%S"),
                "handle": f"@{tweet.user.screen_name}",
                "name": tweet.user.name,
                "oldText": old_text,
                "text": full_text,
                "URL": f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}",
                "type": tweet_type,
                "retweets": tweet.retweet_count,
                "favorites": tweet.favorite_count,
            }

        return outtweets

    def get_complete_tweets_by_ids(self, tweets_ids: list):
        """
        Retrieve a list of completed tweets for tweet ids
        """

        tot_tweets = {}
        for i, tweet_id in enumerate(tweets_ids, start=1):
            tot_tweets[i] = {}
            tot_tweets[i]["id"] = int(tweet_id)
            tot_tweets[i]["fulltext"] = None

        df = pd.DataFrame.from_dict(
            data=tot_tweets, orient="index", columns=["id", "fulltext"]
        )

        lim_start = 100
        start = 0
        final = lim_start + Helpers.count_null_id(tweets_ids, finish=lim_start)
        ids = tweets_ids[start:final]

        iter_needed = math.ceil(
            (len(tweets_ids) - Helpers.count_null_id(tweets_ids)) / 100
        )

        print("Completing tweets..")
        itera = 0
        while itera < iter_needed + 1:
            Helpers.dynamic_text(f"{itera}/{iter_needed}")

            if self.app.debug:
                print(f"{itera}/{iter_needed}")
                # print(f"{start=} {final=}")

            try:
                compl_tweets = self.api.statuses_lookup(id_=ids, tweet_mode="extended")
            except tweepy.TweepError as error:
                if error.api_code == 38:
                    # End of loop
                    break
                print("Error", error)

            # Edit tot_tweets dict
            for compl_tweet in compl_tweets:
                if self.app.debug and compl_tweet.id == 1239689814867312641:
                    print("Hello")
                    print(compl_tweet)

                if hasattr(compl_tweet, "retweeted_status"):
                    # Is RT
                    full_text = compl_tweet.retweeted_status.full_text
                else:
                    # Not a Retweet
                    full_text = compl_tweet.full_text

                df.loc[df["id"] == compl_tweet.id, ["fulltext"]] = full_text

            # Setup request for new iteration
            last_iter_final = final
            start = last_iter_final
            if start + 100 <= len(tweets_ids):
                final = (
                    last_iter_final
                    + 100
                    + Helpers.count_null_id(tweets_ids, start=start, finish=start + 100)
                )
            else:
                final = len(tweets_ids)

            ids = tweets_ids[start:final]
            itera += 1

            if self.app.debug and itera == iter_needed:
                break

        return df


if __name__ == "__main__":
    app = App(debug=True)
    api = Api(app.consumer_key, app.consumer_secret, app.access_key, app.access_secret)

    tweets_ids_xls = Helpers.extract_ids_file("src/resources/data/fr.pkl")
    print(f"Starting ids length: {len(tweets_ids_xls)}")
    completed_tweets = api.get_complete_tweets_by_ids(tweets_ids_xls)
    print(f"Finishing with ids length: {len(completed_tweets)}")
