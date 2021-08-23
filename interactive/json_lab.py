# pylint: skip-file

#%%
import sys, os

sys.path.append(os.path.abspath(os.path.join("..", "src")))
# sys.path.append(os.path.abspath(os.path.join("src")))

import subprocess
import json

from common.app import App

app = App(debug=False)

#%%
# Flatten all jsonl files
jsonl_path = os.path.join(app.root_dir, "database", "jsonl")
jsonl_fs = os.listdir(jsonl_path)

for jsonl_f in jsonl_fs:
    new_jsonl_f = jsonl_f[:-6]  # Remove extension from name
    infile = os.path.join(jsonl_path, jsonl_f)
    outfile = f"{os.path.join(jsonl_path, 'flat', new_jsonl_f)}_flat.jsonl"

    # subprocess.run(
    #     [
    #         "twarc2",
    #         "flatten",
    #         infile,
    #         outfile,
    #     ],
    #     shell=True,
    # )
    print("From", infile)
    print("To", outfile)

# %%

# Read smallest jsonl file to get the hang of it
# jsonl_file = os.path.join(jsonl_path, "brexitparty_uk.jsonl")
# jsonl_file_flat = os.path.join(jsonl_path, "flat", "brexitparty_uk_flat.jsonl")

jsonl_file = os.path.join(jsonl_path, "UDCch.jsonl")
jsonl_file_flat = os.path.join(jsonl_path, "flat", "UDCch_flat.jsonl")

with open(jsonl_file) as jsonl, open(jsonl_file_flat) as jsonl_flat:
    tweets = [json.loads(line) for line in jsonl]
    tweets_flat = [json.loads(line) for line in jsonl_flat]

"""
In non-flat format, each request (containing max 100 tweets) is split.
For UDCch (327 tweets tot.):
len(tweets[0]) == 4
len(tweets[0]["data"]) == 100
tweets[0]["data"][0]["text"] to get text of first tweet

but len(tweets[3]["data"]) == 27
which matches the len(tweets_flat) == 327

"""
#%%
