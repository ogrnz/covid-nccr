# pylint: skip-file

#%%
import sys, os

# sys.path.append(os.path.abspath(os.path.join("..", "src")))
sys.path.append(os.path.abspath(os.path.join("src")))

import subprocess
import json

# import tqdm

from common.app import App
from common.api import Api
from common.database import Database
from common.helpers import Helpers
from common.insertor import InsertFromJsonl

app = App()
api = Api(app)

#%%
jsonl_path = os.path.join(app.root_dir, "database", "jsonl")
handles = ["Sante_Gouv", "Left_EU"]
handles_idx = api.get_user_id_from_handle(handles)

for handle, handle_id in zip(handles, handles_idx):
    outfile = os.path.join(jsonl_path, f"{handle}.jsonl")
    cmd = [
        "twarc2",
        "timeline",
        "--start-time 2020-01-01T00:00:00",
        "--end-time 2021-04-01T00:00:00",
        "--use-search",
        handle,
        outfile,
        # ">> tweets.jsonl",
    ]
    # print(cmd)
    subprocess.run(
        cmd,
        shell=True,
    )

# %%

# twarc2 timeline --start-time 2020-01-01 --end-time 2021-04-01 --use-search Sante_Gouv D:\laragon\www\python\covidProject\database\jsonl\Sante_Gouv.jsonl

# twarc2 timeline --start-time 2020-01-01 --end-time 2021-04-01 --use-search Left_EU D:\laragon\www\python\covidProject\database\jsonl\Left_EU.jsonl
