# pylint: skip-file

"""
Extract unstructured location data into a given category:

"""

#%%
import sys, os

sys.path.append(os.path.abspath(os.path.join("..", "src")))

import re
import time

import pandas as pd
import numpy as np
from tqdm import tqdm

tqdm.pandas()

from common.database import Database
from common.app import App
from common.helpers import Helpers

app_run = App(debug=True)
db = Database("tweets.db", app=app_run)