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
    COUNT = 500  # Academic account

    def __init__(self, main_app: App = None):
        self.app = main_app if not None else App()
        self.connect_api()

    def connect_api(self):
        """
        Connect to twitter API through tweepy
        """

        try:
            auth = tweepy.AppAuthHandler(
                self.app.consumer_key, self.app.consumer_secret
            )
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
            screen_name=screen_name, count=self.COUNT, tweet_mode="extended"
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
                screen_name=screen_name,
                count=self.COUNT,
                max_id=oldest,
                tweet_mode="extended",
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
                    old_text = tweet.full_text
                    prefix = "RY @", tweet.in_reply_to_screen_name, ": "
                    prefix = "".join(prefix)

                    old_text = prefix, old_text
                    old_text = "".join(old_text)

            # Make final dict
            outtweets[index] = {col: None for col in Helpers.schema_cols}
            outtweets[index]["tweet_id"] = tweet.id
            outtweets[index]["created_at"] = tweet.created_at.strftime(
                "%d/%m/%Y %H:%M:%S"
            )
            outtweets[index]["handle"] = f"@{tweet.user.screen_name}"
            outtweets[index]["name"] = tweet.user.name
            outtweets[index]["oldText"] = old_text
            outtweets[index]["text"] = full_text
            outtweets[index][
                "URL"
            ] = f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}"
            outtweets[index]["type"] = tweet_type
            outtweets[index]["retweets"] = tweet.retweet_count
            outtweets[index]["favorites"] = tweet.favorite_count

        return outtweets

    def get_tweets_by_ids(self, tweets_ids: list):
        """
        Retrieve a list of completed (text) tweets for tweet ids
        """

        tot_tweets = {}
        for i, tweet_id in enumerate(tweets_ids, start=1):
            tot_tweets[i] = {col: None for col in Helpers.schema_cols}
            tot_tweets[i]["tweet_id"] = int(tweet_id)

        df = pd.DataFrame.from_dict(
            data=tot_tweets,
            orient="index",
            columns=Helpers.schema_cols,
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

            try:
                compl_tweets = self.api.statuses_lookup(id_=ids, tweet_mode="extended")
            except tweepy.TweepError as error:
                if error.api_code == 38:
                    # End of loop
                    break
                print("Error", error)

            # Edit tot_tweets dict
            for compl_tweet in compl_tweets:
                old_text = None
                if hasattr(compl_tweet, "retweeted_status"):  # Is RT
                    old_text = compl_tweet.full_text
                    tweet_type = "Retweet"
                    full_text = compl_tweet.retweeted_status.full_text
                else:  # Not a Retweet
                    full_text = compl_tweet.full_text
                    tweet_type = "New"

                    if compl_tweet.in_reply_to_status_id is not None:
                        tweet_type = "Reply"
                        old_text = compl_tweet.full_text
                        prefix = "RY @", compl_tweet.in_reply_to_screen_name, ": "
                        prefix = "".join(prefix)

                        old_text = prefix, old_text
                        old_text = "".join(old_text)

                # Create a dict for each retrieved tweet and update all fields of the original df
                new_tweet = {col: None for col in Helpers.schema_cols}
                for coded_col in [
                    "topic",
                    "subcat",
                    "position",
                    "frame",
                    "theme_hardcoded",
                ]:
                    del new_tweet[coded_col]

                # Generate dict of the new tweet
                new_tweet["tweet_id"] = compl_tweet.id
                new_tweet["created_at"] = compl_tweet.created_at.strftime(
                    "%d/%m/%Y %H:%M:%S"
                )
                new_tweet["handle"] = f"@{compl_tweet.user.screen_name}"
                new_tweet["name"] = compl_tweet.user.name
                new_tweet["oldText"] = old_text
                new_tweet["text"] = full_text
                new_tweet[
                    "URL"
                ] = f"https://twitter.com/{compl_tweet.user.screen_name}/status/{compl_tweet.id}"
                new_tweet["type"] = tweet_type
                new_tweet["retweets"] = compl_tweet.retweet_count
                new_tweet["favorites"] = compl_tweet.favorite_count

                # Update df with dict
                Helpers.update_df_with_dict(df, new_tweet)

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

        return df

    def get_tweets_by_ids_with_nan(self, tweets_ids: list, df, no_id_remove=False):
        """
        The idea is to do the same as the get_tweets_by_ids() method,
        but to return a modified df instead of a new one.
        """

        # Otherwise too long to handle
        df["tweet_id"].astype(str)

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

            try:
                compl_tweets = self.api.statuses_lookup(id_=ids, tweet_mode="extended")
            except tweepy.TweepError as error:
                if error.api_code == 38:
                    # End of loop
                    # print("Break loop", error.api_code)
                    break
                print("Error", error)

            # Edit tot_tweets dict
            for compl_tweet in compl_tweets:
                compl_tweet.id = str(compl_tweet.id)
                old_text = None
                if hasattr(compl_tweet, "retweeted_status"):
                    # Is RT
                    old_text = compl_tweet.full_text
                    tweet_type = "Retweet"
                    full_text = compl_tweet.retweeted_status.full_text
                else:  # Not a Retweet
                    full_text = compl_tweet.full_text
                    tweet_type = "New"

                    if compl_tweet.in_reply_to_status_id is not None:
                        tweet_type = "Reply"
                        old_text = compl_tweet.full_text
                        prefix = "RY @", compl_tweet.in_reply_to_screen_name, ": "
                        prefix = "".join(prefix)

                        old_text = prefix, old_text
                        old_text = "".join(old_text)

                new_tweet = {col: None for col in Helpers.schema_cols}
                for coded_col in [
                    "topic",
                    "subcat",
                    "position",
                    "frame",
                    "theme_hardcoded",
                ]:
                    del new_tweet[coded_col]

                # Generate dict of the new tweet
                new_tweet["tweet_id"] = compl_tweet.id
                new_tweet["created_at"] = compl_tweet.created_at.strftime(
                    "%d/%m/%Y %H:%M:%S"
                )
                new_tweet["handle"] = f"@{compl_tweet.user.screen_name}"
                new_tweet["name"] = compl_tweet.user.name
                new_tweet["oldText"] = old_text
                new_tweet["text"] = full_text
                new_tweet[
                    "URL"
                ] = f"https://twitter.com/{compl_tweet.user.screen_name}/status/{compl_tweet.id}"
                new_tweet["type"] = tweet_type
                new_tweet["retweets"] = compl_tweet.retweet_count
                new_tweet["favorites"] = compl_tweet.favorite_count

                # Update df with dict
                Helpers.update_df_with_dict(df, new_tweet)

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

        # Reorder cols
        cols = df.columns.tolist()
        if not no_id_remove:
            cols.remove("tweet_id")
        df = df[cols]

        return df

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

            try:
                compl_tweets = self.api.statuses_lookup(id_=ids, tweet_mode="extended")
            except tweepy.TweepError as error:
                if error.api_code == 38:
                    # End of loop
                    break
                print("Error", error)

            # Edit tot_tweets dict
            for compl_tweet in compl_tweets:
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

        return df


if __name__ == "__main__":
    app = App(debug=True)
    api = Api(app)

    tweets_ids_xls = Helpers.extract_ids_file("src/resources/data/fr.pkl")
    print(f"Starting ids length: {len(tweets_ids_xls)}")
    completed_tweets = api.get_complete_tweets_by_ids(tweets_ids_xls)
    print(f"Finishing with ids length: {len(completed_tweets)}")
