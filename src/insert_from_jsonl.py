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


def main(fname):
    tws = []
    for line in insertor.read(fname):
        tws.append(insertor.ret_tw_from_line(line))

    count = db.insert_many(tws)

    return count


if __name__ == "__main__":
    app = App()
    db = Database("tweets.db", app)
    insertor = InsertFromJsonl(app, [])

    files = os.listdir("./database/jsonl/flat/")
    files = [fname for fname in files if fname.endswith("jsonl")]

    with db:
        tot_count = 0
        for f in tqdm(files):
            tot_count += main(f)
    print(f"{tot_count} tweets inserted")
