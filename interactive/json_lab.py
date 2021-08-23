# pylint: skip-file

#%%
import sys, os

sys.path.append(os.path.abspath(os.path.join("src")))

import subprocess
import json

from common.app import App

app = App(debug=False)

#%%
# Flatten all jsonl files
jsonl_path = os.path.join(app.root_dir, "database", "jsonl")
jsonl_fs = os.listdir(jsonl_path)

for jsonl_f in jsonl_fs:
    new_jsonl_f = jsonl_f.strip(".jsonl")
    infile = os.path.join(jsonl_path, jsonl_f)
    outfile = f"{os.path.join(jsonl_path, 'flat', new_jsonl_f)}_flat.jsonl"

    subprocess.run(
        [
            "twarc2",
            "flatten",
            infile,
            outfile,
        ],
        shell=True,
    )
    # print("From", infile)
    # print("To", outfile)

# %%
