# pylint: skip-file
"""
Random interactive debug file.
"""

#%%
import sys, os

sys.path.append(os.path.abspath(os.path.join("..", "src")))

#%%
import pandas as pd
import numpy as np

from common.database import Database
from common.app import App
from common.api import Api

app_run = App(debug=True)
db = Database("tweets.db", app=app_run)

#%%
with db:
    tweets = db.get_all_tweets()
len(tweets)

# %%
df = pd.DataFrame(
    tweets,
    columns=[
        "tweet_id",
        "covid_theme",
        "created_at",
        "handle",
        "name",
        "oldText",
        "text",
        "URL",
        "type",
        "retweets",
        "favorites",
        "topic",
        "subcat",
        "position",
        "frame",
        "theme_hardcoded",
    ],
)
# %%
df[df["tweet_id"] == "1220281260838395910"]["type"] == "Reply"  # True
df[df["tweet_id"] == "1224640608876273665"]["type"] == "Reply"  # False
# what's wrong with types

#%%
"""
Some replies and retweets have an extra space
"""
df["type"].unique()
# array(['Retweet', 'New', 'Reply', 'Retweet ', 'Reply '], dtype=object)
# %%
df["type"] = df["type"].str.replace(" ", "")
df["type"].unique()
# %%
len(df[df["type"] == "Reply"])
# yikes

#%%
to_update = [
    (
        tweet[8],
        tweet[0],
    )
    for i, tweet in enumerate(df.values.tolist())
]
print(len(to_update))

# %%
with db:
    count = db.update_many("type", "tweet_id", to_update)

print(count, "tweets updated")

# %%
