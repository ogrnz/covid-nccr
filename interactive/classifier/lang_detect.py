"""
Detect language of tweets using fasttext
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
from common.lang_detect import fasttext_detect, fasttext_lang_detect, fasttext_en_detect

tqdm.pandas()

#%%
app_run = App(debug=False)
db = Database("tweets.db", app=app_run)
data_path = os.path.join(app_run.root_dir, "interactive", "data")
lang_path = os.path.join(data_path, "lang")

#%%
with db:
	tweets = db.get_all_tweets()
len(tweets)

# %%
df = Helpers.df_from_db(tweets)

# %%
df_samp = df.sample(5000)
len(df_samp)

# %%
def preprocess_lang(txt):
	return "other" if txt not in ["en", "fr"] else txt

#%%
# lang only for those 5000
t1 = perf_counter()
df_samp["lang"] = df_samp["text"].apply(fasttext_lang_detect)
print(perf_counter() - t1, "s")
# check df_samp["lang"]

# %%
df_samp["lang"] = df_samp["lang"].apply(preprocess_lang)

"""
en       3018
fr       1616
other     366
"""

# %%
# For whole database
t1 = perf_counter()
df["lang"] = df["text"].progress_apply(fasttext_lang_detect)
print(perf_counter() - t1, "s")

df["lang"].value_counts()

# %%
df["lang"] = df["lang"].apply(preprocess_lang)
df["lang"].value_counts()

"""
en       139238, 0.58
fr        80173, 0.33
other     19112, 0.08
"""

# %%
# Save en tweets to pickle
df[df["lang"] == "en"].to_pickle(os.path.join(lang_path, "db_en.pkl"))
df[df["lang"] == "fr"].to_pickle(os.path.join(lang_path, "db_fr.pkl"))
df[df["lang"] == "other"].to_pickle(os.path.join(lang_path, "db_other.pkl"))
