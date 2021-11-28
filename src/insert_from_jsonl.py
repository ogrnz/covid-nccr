"""
Import jsonl FLATTENED file into the database.
Already present tweets are updated.
"""

import os
import logging

# import time

from tqdm import tqdm

from common.app import App
from common.database import Database

# from common.helpers import Helpers
from common.insertor import InsertFromJsonl

log = logging.getLogger(os.path.basename(__file__))


def main(fname, mode):
    tws = [insertor.ret_tw_from_line(line) for line in insertor.read(fname)]
    return (
        sum(1 for tw in tqdm(tws) if db.insert_no_ignore(tw))
        if mode == "single"
        else db.insert_many(tws)
    )


if __name__ == "__main__":
    app = App()
    db = Database("tweets.db", app)
    insertor = InsertFromJsonl(app, [])

    # files = os.listdir("./database/jsonl/flat/")
    # files = [fname for fname in files if fname.endswith("jsonl")]
    # files = ["Sante_Gouv_flat.jsonl", "Left_EU_flat.jsonl"]
    files = ["Sante_Gouv_flat.jsonl"]

    with db:
        tot_count = 0
        for f in tqdm(files):
            tot_count += main(f, mode="single")

    log.info(f"{tot_count} tweets inserted")
