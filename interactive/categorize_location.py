# pylint: skip-file

"""
Extract unstructured location data into a given category:
- International (international OI, not EU)
- Europe (european OI/ONG)
- "country_name": when handle only comes from a country and it does not work in any of the 2 mentioned OI/ONG.
"""

#%%
%load_ext autoreload
%autoreload 

#%%
import sys, os

sys.path.append(os.path.abspath(os.path.join("..", "src")))

import csv
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

#%%
import pycountry

#%%
# Import csv

info = []
csv_path = os.path.join(app_run.root_dir, "src", "resources", "agents.csv")
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        info.append(row)
print(info)

# %%
for user in tqdm(info):
    user["location_category"] = None
    for country in pycountry.countries:
        if country.name in user["location"]:
            user["location_category"] = country.name

#%%
for user in tqdm(info):
    if user["location"] == "NA":
        user["location_category"] = "NA"
    if user["TweepError"] != "":
        user["location_category"] = "TweepError"

# %%
na_count = 0

for user in info:
   if user["location_category"] is None:
       na_count += 1 

print(na_count)  # 4765 

# %%
csv_path = os.path.join(app_run.root_dir, "src", "resources", "agents_cat.csv")
with open(csv_path, "w", encoding="utf-8") as f:
    fields = list(info[0].keys())
    writer = csv.DictWriter(f, delimiter=",", fieldnames=fields)
    writer.writeheader()

    for user in info:
        writer.writerow(user) 
