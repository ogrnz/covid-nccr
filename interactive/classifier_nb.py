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
df_other = pd.read_pickle("data/db_other.pkl")

#%%
# Only keep already coded tweets (6XX or "theme_hardcoded == 0")
sub = df[~df["topic"].isnull()]
