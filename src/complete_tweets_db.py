# pylint: skip-file

"""
Retrieve extended and complete tweet, write it to "text" column and keep the old with "RT" and "RY" in "old_text".
"""

#%%
%load_ext autoreload
%autoreload 2

import os
import logging

import pandas as pd
import tqdm

from common.database import Database
from common.app import App
from common.api import Api
from common.helpers import Helpers

app_run = App(debug=True)
db = Database("tweets.db", app=app_run)
api = Api(app_run)
log = logging.getLogger(os.path.basename(__file__))


#%%
with db:
    tws = db.get_all_tweets(("text", None))
print(len(tws))

# %%
tws_idx = [tw[0] for tw in tws]

# !! Delete tweet_ids are less than len(19)
# Those were manually generated and do not correspond to real tweets with those ids.
not_idx = [id for id in tws_idx if len(id) < 19]

for not_id in not_idx:
    tws_idx.remove(not_id)
len(tws_idx)  # should be len(tws_idx) - len(not_idx) = 240_375 - 1_019 = 239_356

# %%
tws_completed = api.get_complete_tweets_by_ids(tws_idx)

# %%
# Get those completed tweets and update db by id with new text
to_update = [
    (
        tweet[1],
        tweet[0],
    )
    for i, tweet in enumerate(tws_completed.values.tolist())
]

# %%
with db:
    count = db.update_many("text", "tweet_id", to_update)

log.info(count, "tweets updated")
# check 1216694717288632320, 1258741133326385153, 1220705634074624000
#ok

# %%
# Cause rate limit
# for tw_id in tws_idx:
#     api.get_complete_tweets_by_ids([tw_id])

# %%
