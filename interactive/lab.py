# pylint: skip-file
"""
Debug
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
db = Database("lab.db", app=app_run)

# %%
# Connect to the API
api = Api(
    app_run.consumer_key,
    app_run.consumer_secret,
    app_run.access_key,
    app_run.access_secret,
    main_app=app_run,
)

# %%
