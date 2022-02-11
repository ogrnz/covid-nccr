"""
Mark all english tweets as such
"""

#%%
import sys, os

sys.path.append(os.path.abspath(os.path.join("..", "..", "src")))

from time import perf_counter
import pandas as pd
from tqdm import tqdm

from common.database import Database
from common.app import App
from common.helpers import Helpers
from common.lang_detect import fasttext_lang_detect, fasttext_en_detect

tqdm.pandas()

#%%
app_run = App(debug=False)
db = Database("tweets.db", app=app_run)
data_path = os.path.join(app_run.root_dir, "interactive", "data")

#%%
with db:
	tweets = db.get_all_tweets()
len(tweets)

# %%
df = Helpers.df_from_db(tweets)

# %%
df_samp = df.sample(5000)
len(df_samp)

#%%
# en_detect only for those 5000
t1 = perf_counter()
df_samp["en_detect"] = df_samp["text"].apply(fasttext_en_detect)
print(perf_counter() - t1, "s")
sum(df_samp["en_detect"])  # 139238 seems correct

# %%
# For whole database
t1 = perf_counter()
df["en_detect"] = df["text"].progress_apply(fasttext_en_detect)
print(perf_counter() - t1, "s")
sum(df["en_detect"])

# %%
# Save en tweets to pickle
df.to_pickle(os.path.join(data_path, "db_en_detect.pkl"))