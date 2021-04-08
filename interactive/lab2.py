#%%

import tweepy
import pandas as pd

from common.database import Database
from common.app import App
from common.helpers import Helpers
from common.api import Api

app_run = App(debug=True)
database = Database("tweets_tests.db", app=app_run)

# %%
# Connect to the API
api = Api(
    app_run.consumer_key,
    app_run.consumer_secret,
    app_run.access_key,
    app_run.access_secret,
    main_app=app_run,
)

# %%
ids = [1263097241465769985, 1283438098391728129]
# ids = [1263097241465769985]
tweets = api.api.statuses_lookup(id_=ids, tweet_mode="extended")

#%%
for tweet in tweets:
    print(tweet.in_reply_to_screen_name)

# %%
