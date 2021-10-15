# pylint: skip-file

"""
This file is used to retrieve information about some users and write it to a .csv file.
At the moment, the user_id, location, verified status, followers count are retrieved.
"""

#%%
import sys, os

sys.path.append(os.path.abspath(os.path.join("..", "src")))

#%%
%load_ext autoreload
%autoreload 2

#%%
import json
import csv
import time

import pandas as pd
import numpy as np
from tqdm import tqdm
import tweepy

from common.app import App
from common.api import Api
from common.helpers import Helpers

app_run = App(debug=False)
api = Api(app_run)

# %%
f_name = "all_actors_sorted_cleaned.txt"
f_path = os.path.join(app_run.root_dir, "src", "resources", f_name)
with open(f_path, "r") as f:
    actors = f.readlines()

print("Total list length: ", len(actors))
actors = [actor.lower() for actor in actors]
actors = set(actors)
print("Unique values counts: ", len(actors))
actors = [actor.strip("\n") for actor in actors]
actors.sort()

# %%
with open(os.path.join(app_run.root_dir, "src", "resources", "all_actors_sorted.txt"), "w", encoding="utf-8") as f:
    for actor in actors:
        f.write(f"{actor}\n")

# %%
retrieved_users = api.get_many_users(actors)

#%%
tot = 0
for req in retrieved_users:
    for user in req:
        tot += 1
print(f"Got {tot} users. {len(actors) - tot} users were not found.")

#%%
users_dict = {handle:None for handle in actors}

for req in retrieved_users:
    for user in req:
        handle = user.screen_name
        location = user.location if user.location != "" else "NA"

        users_dict[handle.lower()] = {"user_id": user.id, "location": location, "followers_count": user.followers_count,  "verified": user.verified, "TweepError": None}

# %%
def get_info(handles):
    """
    This function is here to get individual information on users and especially to know the reason why the precendent bulk request didn't work.
    """

    users = dict()
    for i, handle in enumerate(tqdm(handles)):
        last_id = i
        try:
            user = api.get_user(handle)
            location = user.location if user.location != "" else "NA"

            users[handle.lower()] = {"user_id": user.id, "location": location, "followers_count": user.followers_count,  "verified": user.verified, "TweepError": None}

        except tweepy.TweepError as tweep_e:
            api_code = tweep_e.args[0][0]["code"]
            print("TweepError", tweep_e)

            users[handle.lower()] = {"user_id": None, "location": None, "followers_count": None,  "verified": None, "TweepError": api_code}

            if api_code == 88:
                print(f"RateLimitError handle {handle} ({i})")
                return users

        except KeyboardInterrupt:
            print(f"{last_id=}")
            return users

    return users

#%%
# For those problematic users, doublecheck and insert them into csv with reason

# Get the names of users which were not retrieved
not_found_users = [k for k in users_dict.keys() if users_dict[k] is None]
len(not_found_users)
# 447 users

#%%
prob_users = get_info(not_found_users)  # 145
len(prob_users)

#%%
# Complete users_dict with individually gotten user info
for prob_user in prob_users.items():
    users_dict[prob_user[0]] = prob_user[1]

# !!! Remember to "users_dict["last_user"] = None" if the request was stopped because of RateLimit !!!

#%%
# Note:
# TweepError code: 50 -> user not found (account deleted)
# TweepError code: 63 -> user suspended

#%%
to_write = []
for user in users_dict.items():
    user_info = (user[0], user[1]["user_id"], user[1]["location"], user[1]["followers_count"], user[1]["verified"], user[1]["TweepError"])
    to_write.append(user_info)
print("Do actors an to_write have same length?", len(actors) == len(to_write))

# %%
# write to file
csv_path = os.path.join(app_run.root_dir, "src", "resources", "agents2.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as csv_f:
    writer = csv.writer(csv_f, delimiter="\t")
    writer.writerow(["handle", "user_id", "location", "followers_count", "verified", "TweepError"])
    for user in to_write:
        writer.writerow(user)

# %%
