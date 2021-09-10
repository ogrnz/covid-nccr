"""
Import jsonl FLATTENED file into the database.
Already present tweets are updated.
"""

import os

# import time

from tqdm import tqdm

from common.app import App
from common.database import Database

# from common.helpers import Helpers
from common.insertor import InsertFromJsonl


def main(fname, mode):
    tws = []
    for line in insertor.read(fname):
        tws.append(insertor.ret_tw_from_line(line))

    if mode == "single":
        # To know which tweet triggers the unique constraint
        count = 0
        for tw in tqdm(tws):
            if db.insert_no_ignore(tw):
                count += 1
    else:
        count = db.insert_many(tws)

    return count


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

    print(f"{tot_count} tweets inserted")
