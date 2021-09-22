# pylint: skip-file

"""
Another random experimentation file.
"""

#%%
import sys, os

sys.path.append(os.path.abspath(os.path.join("..", "src")))

#%%
import json

import pandas as pd
import numpy as np
from tqdm import tqdm

from common.database import Database
from common.app import App
from common.api import Api
from common.helpers import Helpers
from common.insertor import InsertFromJsonl

app_run = App(debug=True)
db = Database("tweets.db", app=app_run)

# %%
# Connect to the API
api = Api(app_run)

# %%
with db:
    tws = db.get_all_tweets(condition=("handle", "@Sante_Gouv"))

insertor = InsertFromJsonl(app_run, [])
jsonl_path = os.path.join(app_run.root_dir, "database", "jsonl")
test_file = "Sante_Gouv_flat.jsonl"
jsonl_file_flat = os.path.join(jsonl_path, "flat", test_file)

with open(jsonl_file_flat) as jsonl_flat:
    tws_flat = [json.loads(line) for line in jsonl_flat]

# %%
tws_flat_idx = [line["id"] for line in tws_flat]
df = Helpers.df_from_db(tws)

# %%
df_flat = pd.DataFrame(tws_flat_idx, columns={"tweet_id"})
missing_idx = df_flat[
    ~df_flat["tweet_id"].isin(df["tweet_id"])
].values.tolist()  # those tweets were not inserted in db. Why?
missing_idx = [x[0] for x in missing_idx]

# %%
with db:
    tws_check = []
    for tw_id in tqdm(missing_idx):
        tws_check.append(db.get_tweet_by_id(tw_id))

# %%
names = set()
for tw in tws_check:
    names.add(tw[0][3])

print(names)
# All the other tweets were already in the Database, but under @MinSoliSante!
# %%
