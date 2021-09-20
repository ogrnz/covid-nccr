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
f_path = os.path.join(app_run.root_dir, "src", "resources", "all_actors.txt")
with open(f_path, "r") as f:
    actors = f.readlines()

print("Total list length: ", len(actors))
actors = set(actors)
print("Unique values counts: ", len(actors))
actors = [actor.strip("\n") for actor in actors]
actors.sort()

# %%
with open(os.path.join(app_run.root_dir, "src", "resources", "all_actors_sorted.txt"), "w", encoding="utf-8") as f:
    for actor in actors:
        f.write(f"{actor}\n")

# %%
users = api.get_many_users(actors)

#%%
tot = 0
for req in users:
    for user in req:
        tot += 1
print(f"Got {tot} users. {len(actors) - tot} users were not found.")

#%%
retrieved_users = set()

# with open(csv_path, "a", newline="", encoding="utf-8") as csv_f:
    # writer = csv.writer(csv_f, delimiter="\t", quotechar="|", quoting=csv.QUOTE_MINIMAL)

    # writer.writerow(["handle", "user_id", "location", "verified"])

for req in tqdm(users):
    for user in req:
        handle = user.screen_name
        tw_id = user.id
        location = user.location if user.location != "" else "NA"
        verified = user.verified
        retrieved_users.add((handle, tw_id, location, verified))

        # writer.writerow([handle, tw_id, location, verified])
# %%
def get_info(handles):
    # with open(csv_path, "a", newline="", encoding="utf-8") as csv_f:
    #     writer = csv.writer(csv_f, delimiter="\t", quotechar="|", quoting=csv.QUOTE_MINIMAL)

    users = set()
    for i, handle in enumerate(tqdm(handles)):
        try:
            user = api.get_user(handle)
            tw_id = user.id
            location = user.location if user.location != "" else "NA"
            verified = user.verified

            users.add((handle, tw_id, location, verified))

        except tweepy.TweepError as tweep_e:
            api_code = tweep_e.args[0][0]["code"]
            print("TweepError", tweep_e)
            tw_id = f"TweepError code: {api_code}"
            location, verified = [ "NA" for _ in range(2)]

            if api_code == 88:
                print(f"RateLimitError handle {handle} ({i})")
                last_id = i
                break

        except KeyboardInterrupt:
            print(f"{last_id=}")
            exit()

        # writer.writerow([handle, tw_id, location, verified])
    return users

#%%
# For those problematic users, doublecheck and insert them into csv with reason

# Get the names of users which were not retrieved
# not_found_users = api.get_user_id_from_handle(actors)
retrieved_handles = [user[0] for user in retrieved_users]
not_found_users = set(actors) - set(retrieved_handles)  # 248

prob_users = get_info(not_found_users)  # 142

#%%
retrieved_users_idx = set(user[1] for user in retrieved_users)  # 2444
prob_users_idx = set(user[1] for user in prob_users)  # 137
all_idx = retrieved_users_idx.union(prob_users_idx)

users = api.get_many_users(list(all_idx), mode="user_ids")  # 2444

# Note:
# TweepError code: 50 -> user not found (account deleted)
# TweepError code: 63 -> user suspended

#%%
to_write = []
for req in tqdm(users):
    for user in req:
        handle = user.screen_name
        tw_id = user.id
        location = user.location if user.location != "" else "NA"
        verified = user.verified
        to_write.append((handle, tw_id, location, verified))

# %%
# write to file

csv_path = os.path.join(app_run.root_dir, "src", "resources", "agents.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as csv_f:
    writer = csv.writer(csv_f, delimiter="\t")
    writer.writerow(["handle", "user_id", "location", "verified"])
    for user in to_write:
        writer.writerow(user)

# %%
