# pylint: skip-file
#%%
import sys, os

sys.path.append(os.path.abspath(os.path.join("..", "src")))

#%%
import pandas as pd

from tqdm import tqdm

from common.database import Database
from common.app import App

#%%
df_en = pd.read_pickle("data/db_en.pkl")
df_fr = pd.read_pickle("data/db_fr.pkl")
df_other = pd.read_pickle("data/db_other.pkl")

#%%
""" ENG """
# Drop a few cols for readability
df_en.drop(
    [
        "covid_theme",
        "created_at",
        "handle",
        "name",
        "URL",
        "type",
        "retweets",
        "favorites",
        "position",
        "frame",
    ],
    axis=1,
    inplace=True,
)

#%%
def covid_classify(row: pd.Series):
    if (
        row["topic"] == "608"
        or row["topic"] == "608.0"
        or row["theme_hardcoded"] == "0"
    ):
        return 0
    return 1


#%%
# Only keep already coded tweets (6XX or "theme_hardcoded == 0")
df_en = df_en[~df_en["topic"].isnull() | ~df_en["theme_hardcoded"].isnull()]

# If 608 or theme_hardcoded == 0, then set new col `y` = 0
df_en["y"] = df_en.apply(covid_classify, axis=1)

# %%
X = pd.DataFrame({"text": []})
X["text"] = df_en.apply(
    lambda r: r["oldText"] if r["text"] is None else r["text"], axis=1
)
y = df_en["y"]

# %%
