# pylint: skip-file
"""
Lab interactive file
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
from common.helpers import Helpers

app_run = App(debug=True)
db = Database("tweets.db", app=app_run)

# %%
# Connect to the API
api = Api(app_run)

# %%
with db:
    tweets = db.get_all_tweets()
len(tweets)

# %%
df = Helpers.df_from_db(tweets)

# %%

# %%
