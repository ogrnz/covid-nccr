from datetime import date
from datetime import datetime as dt

import tweepy

from  src.common.helpers import Helpers

class Api:  
    """
    Handles communication with the Twitter API through
    tweepy
    """

    def __init__(self):
        pass

    def connect_api(self, cons_key, cons_scrt, 
            acc_key, acc_scrt):
        try:
            auth = tweepy.OAuthHandler(cons_key, cons_scrt)
            auth.set_access_token(acc_key, acc_scrt)
            self.api = tweepy.API(auth)

            return self.api
        except Exception as e:
            return e

    def get_tweets(self, screen_name, last_id):
        """
        Retrieve tweets from a user 
        Script from @yanofsky as baseline
        https://gist.github.com/yanofsky/5436496
        """

        alltweets = []
        new_tweets = self.api.user_timeline(
            screen_name=screen_name, count=200, 
            tweet_mode="extended"
        )
        alltweets.extend(new_tweets)
        oldest = alltweets[-1].id - 1

        while len(new_tweets) > 0:
            # If tweet older than that ID (== 31/12/2019)
            # or older than last ID in db for that actor, go to next actor
            if oldest < 1211913001147740161 or oldest < last_id:
                break

            print(f"Getting tweets before {oldest}")

            new_tweets = self.api.user_timeline(
                screen_name=screen_name, 
                count=200, max_id=oldest,
                tweet_mode="extended"
            )
            alltweets.extend(new_tweets)
            oldest = alltweets[-1].id - 1
            
            Helpers.dynamic_text(f"...{len(alltweets)} tweets downloaded")

        outtweets = {}
        for index, tweet in enumerate(alltweets):
            # Ignore tweets older than 2019/12/31
            as_of = dt.strptime("2019/12/31", "%Y/%m/%d")
            if tweet.created_at < as_of:
                continue  

            old_text = None
            if hasattr(tweet, 'retweeted_status'): #Is RT
                old_text = tweet.full_text
                tweet_type = 'Retweet'
                full_text = tweet.retweeted_status.full_text
            else:  # Not a Retweet
                full_text = tweet.full_text
                tweet_type = 'New'
                if tweet.in_reply_to_status_id is not None:
                    tweet_type = 'Reply'

            # Make final dict
            outtweets[index] = {
                'tweet_id': tweet.id,
                'covid_theme': None, 
                'type': tweet_type,
                'created_at': tweet.created_at.strftime('%d/%m/%Y %H:%M:%S'), 
                'handle': f"@{tweet.user.screen_name}",
                'name': tweet.user.name,
                'oldText': old_text,
                'text': full_text,
                'URL': f'https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}',
                'retweets': tweet.retweet_count,
                'favorites': tweet.favorite_count
            }

            return outtweets

if __name__ == "__main__":
    from src.common.app import App
    
    app = App(debug=True)
    api = Api.connect_api(App.BEARER_TOKEN, App.CONSUMER_KEY,
        App.CONSUMER_SECRET, App.ACCESS_KEY, App.ACCESS_SECRET)