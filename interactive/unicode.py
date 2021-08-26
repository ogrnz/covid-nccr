# pylint: skip-file

#%%
import sys, os

sys.path.append(os.path.abspath(os.path.join("..", "src")))

import pandas as pd
import numpy as np

from common.database import Database
from common.app import App
from common.api import Api
from common.helpers import Helpers

app_run = App(debug=True)
db = Database("test.db", app=app_run)

# %%
tweet = u"\ud83d\udd34 LIVE now: #Health Council has started. EU Ministers discuss the current situation &amp; measures already taken in relation to #coronavirus #covid19 Watch here \ud83d\udc47 #EPSCO \ud83d\udd0d Latest updates about the meeting: https://t.co/0MpHGrtYTv https://t.co/XK9TYTy24W"

# %%
