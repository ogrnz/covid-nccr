"""
API module
"""

from datetime import datetime as dt

import tweepy


class Api:
    """
    Handles communication with the Twitter API through
    tweepy
    """

    api = None
    connected = False

    def __init__(self, CONSUMER_KEY, CONSUMER_SECRET, ACCESS_KEY, ACCESS_SECRET):
        self.connect_api(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_KEY, ACCESS_SECRET)

    def connect_api(self, cons_key, cons_scrt, acc_key, acc_scrt):
        """
        Connect to twitter API through tweepy
        """

        try:
            auth = tweepy.OAuthHandler(cons_key, cons_scrt)
            auth.set_access_token(acc_key, acc_scrt)
            self.api = tweepy.API(auth)
            self.connected = True
        except Exception as error:
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
            outtweets[index] = {
                "tweet_id": tweet.id,
                "covid_theme": None,
                "type": tweet_type,
                "created_at": tweet.created_at.strftime("%d/%m/%Y %H:%M:%S"),
                "handle": f"@{tweet.user.screen_name}",
                "name": tweet.user.name,
                "oldText": old_text,
                "text": full_text,
                "URL": f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}",
                "retweets": tweet.retweet_count,
                "favorites": tweet.favorite_count,
            }

        return outtweets


if __name__ == "__main__":
    from app import App

    app = App(debug=True)
    api = Api(app.CONSUMER_KEY, app.CONSUMER_SECRET, app.ACCESS_KEY, app.ACCESS_SECRET)

    tweets = api.get_tweets("elonmusk")
