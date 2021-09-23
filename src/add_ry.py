# pylint: skip-file

"""
Add "RY @handle: ..." to all replies
"""

#%%
import pandas as pd

from common.database import Database
from common.app import App
from common.api import Api
from common.helpers import Helpers

app_run = App(debug=True)
db = Database("tweets.db", app=app_run)

# %%
# Connect to the API
api = Api()

# %%
with db:
    tweets = db.get_by_type("Reply")
len(tweets)

#%%
df = pd.DataFrame(
    tweets,
    columns=Helpers.schema_cols,
)
df

#%%
tweets_ids = [tweet[0] for tweet in tweets]
len(tweets_ids)
tweets_ids

#%%
tot_tweets = None
tot_tweets = api.get_tweets_by_ids_with_nan(tweets_ids, df, no_id_remove=True)

#%%
print(len(tot_tweets))
print(len(tweets_ids))

to_update = [
    (
        tweet[4],
        tweets_ids[i],
    )
    for i, tweet in enumerate(tot_tweets.values.tolist())
]
print(len(to_update))

# %%
with db:
    count = db.update_many("oldtext", "tweet_id", to_update)

print(count, "tweets updated")


#%%
import pickle

with open("src/resources/data/pkl/tweets.pkl", "wb") as out:
    pickle.dump(tot_tweets, out, pickle.HIGHEST_PROTOCOL)
    pickle.dump(to_update, out, pickle.HIGHEST_PROTOCOL)

# %%

with open("src/resources/data/pkl/tweets.pkl", "rb") as inp:
    tot_tweets = pickle.load(inp)
    to_update = pickle.load(inp)

# %%
len(tot_tweets) - sum(tot_tweets["oldText"].apply(lambda x: str(x).startswith("RY")))
# = 16 tweets with an issue. Most of them answer to deleted tweets
