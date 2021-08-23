# pylint: skip-file

import sys, os

sys.path.append(os.path.abspath(os.path.join("src")))

import subprocess
import time

from common.app import App

app = App(debug=False)
res_path = os.path.join(app.root_dir, "src", "resources")
twarc_path = os.path.join(app.root_dir, "src", "misc", "twarc")

# F_HANDLES = f"{res_path}/actors_handles.txt"
# F_IDX = f"{res_path}/actors_idx.txt"
# START = "2019-12-31T00:00:01"
# END = "2021-03-31T23:59:59"

# For testing purposes
F_HANDLES = os.path.join(res_path, "actors_stripped_handles.txt")
F_IDX = os.path.join(res_path, "actors_stripped_idx.txt")
START = "2021-01-01T00:00:01"
END = "2021-03-31T23:59:59"

with open(F_HANDLES, "r") as f_handles, open(F_IDX) as f_idx:
    handles = f_handles.readlines()
    idx = f_idx.readlines()

handles = [handle.strip("\n") for handle in handles]
idx = [actor_id.strip("\n") for actor_id in idx]

start = time.time()

for handle, actor_id in zip(handles, idx):
    print(handle, actor_id)
    code = subprocess.run(
        [
            "twarc2",
            "timeline",
            "--use-search",
            "--start-time",
            START,
            "--end-time",
            END,
            actor_id,
            f"{os.path.join(twarc_path, 'tweets_data', handle)}.jsonl",
        ],
        shell=True,
    )
    # print(code)

print(f"Finished in {round((time.time() - start)/60, 2)}min")
