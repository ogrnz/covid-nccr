"""
Import jsonl FLATTENED file into the database.
Already present tweets are updated.
"""

import time
import json

from tqdm import tqdm

from common.app import App
from common.api import Api
from common.database import Database
from common.helpers import Helpers
from common.classify import Classifier
from common.insertor import InsertFromJsonl


def main():
    app = App()
    api = Api(app)
    db = Database("tweets.db", app)

    # TEST file
    test_file = "UN_flat_test.jsonl"
    insertor = InsertFromJsonl(app)
    num_tweets = insertor.get_tot_lines(test_file)

    with db:
        tws_db = db.get_all_tweets()
    tws_flat = insertor.read(test_file)

    # Absolutely not optimized, O(n^2), to be improved if possible.
    to_update = []
    for tw_flat in tqdm(tws_flat, total=num_tweets):
        for tw_db in tws_db:
            old_text = (
                tw_db[5][:140].replace("\n", "").replace(" ", "")
                if tw_db[5] is not None
                else None
            )
            text = (
                tw_db[6][:140].replace("\n", "").replace(" ", "")
                if tw_db[6] is not None
                else None
            )
            old_id = tw_db[0]
            old_url = tw_db[7]
            flat_txt = tw_flat["text"][:140].replace("\n", "").replace(" ", "")

            if old_url == "0" and flat_txt in (old_text, text):

                new_id = tw_flat["id"]
                new_url = Helpers.build_tweet_url(new_id, tw_flat["author"]["username"])
                new_created_at = Helpers.twitter_to_db_time(tw_flat["created_at"])

                new_tweet = (new_id, new_url, new_created_at, old_id)

                to_update.append(new_tweet)
                # print(new_tweet)

    print(to_update)
    print(len(to_update))


if __name__ == "__main__":
    main()
