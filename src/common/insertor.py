from abc import ABC, abstractmethod
import os
import os.path
import logging
import json
import unicodedata
import re
from html import unescape

# from common.app import App
# from common.api import Api
# from common.database import Database
from common.helpers import Helpers

log = logging.getLogger(os.path.basename(__file__))


class Insertor(ABC):
    """
    Insertor abstract class (insert_from_xxx)
    """

    @abstractmethod
    def read(self, filename):
        pass

    @abstractmethod
    def preprocess(self, txt):
        pass

    @abstractmethod
    def insert(self):
        pass


class InsertFromJsonl(Insertor):
    """
    Insert from .jsonl files. In this context, those files are generated by the twarc module.
    This class handles FLATTENED (1 tweet per line) jsonl files (see twarc documentation).
    Mode can be "serial" or "multiproc" to handle multiprocessing.
    """

    def __init__(self, app, tws_db: list, mode: str = "serial"):
        self.app = app
        self.jsonl_path = os.path.join(self.app.root_dir, "database", "jsonl", "flat")
        self.tws = tws_db
        self.idx_found = []
        self.mode = mode
        self.to_insert = []

    def read(self, filename):
        """
        Open and extract each line with this generator.
        """

        with open(
            os.path.join(self.jsonl_path, filename),
            encoding="utf8",
        ) as open_f:
            for line in open_f:
                yield json.loads(line)

    def get_tot_lines(self, filename):
        """
        Return the total number of lines of a file.
        """

        with open(
            os.path.join(self.jsonl_path, filename),
            encoding="utf8",
        ) as open_f:
            tot_lines = sum(1 for _ in open_f)
        return tot_lines

    def ret_tw_from_line(self, line):
        """
        Return a tweet tuple in db's format from a flattened jsonl file.
        """

        tweet_type = Helpers.get_type_from_json(line)
        old_text = None
        txt = line["text"]

        if tweet_type in ("Retweet", "Reply"):
            old_text = line["text"]
            txt = None

        tweet = (
            line["id"],  # tweet_id
            1,  # covid_theme
            Helpers.twitter_to_db_time(line["created_at"]),  # created_at
            "@" + line["author"]["username"],  # handle
            line["author"]["name"],  # name
            old_text,  # old_text
            txt,  # text
            Helpers.build_tweet_url(line["id"], line["author"]["username"]),  # url
            tweet_type,  # type
            line["public_metrics"]["retweet_count"],  # retweets
            line["public_metrics"]["like_count"],  # favorites
            None,  # topic
            None,  # subcat
            None,  # position
            None,  # frame
            None,  # theme_hardcoded
        )

        # Small check to prevent breaking the function if we change the number of cols.
        if len(tweet) != len(Helpers.schema_cols):
            log.warning(
                "ValueError: number of columns in function does not match schema."
            )
            raise ValueError("Number of columns in function does not match schema")

        return tweet

    def preprocess(self, txt: str):
        """
        Sanitize a string for the specific needs of this insertor.
        """

        if txt is None:
            return None

        # Replace and format values
        regex = r"(\xa0\w{3}\s\d{2}.\s\d{4}\xa0)"  #  "\xa0Mar 03, 2020\xa0"
        txt = re.sub(regex, "", txt, 1)
        txt = unicodedata.normalize("NFKD", txt)
        txt = unescape(txt)  # for & and >
        txt = txt.replace("\n", "").replace(" ", "").replace("’", "'")
        # txt = txt.replace("&amp;", "&")

        # return txt[:100]
        # return txt[:140]
        return txt

    def check_in_db(self, tw_flat, tws_db: list = None):
        """
        This method checks if the provided jsonl_flat tweet is present in the db.

        Can provide a tws_db value for debugging purpose.
        """

        tws = self.tws
        if tws_db is not None:
            tws = tws_db

        idx_found = self.idx_found
        if self.mode == "multiproc":
            idx_found = []

        new_tweet = None
        found_count = 0
        for tw_db in tws:
            old_id = tw_db[0]
            old_text = self.preprocess(tw_db[5])
            text = self.preprocess(tw_db[6])
            old_url = tw_db[7]
            flat_txt = self.preprocess(tw_flat["text"])

            if old_url == "0" or old_url is None and flat_txt in (old_text, text):
                found_count += 1
                idx_found.append(old_id)
                new_id = tw_flat["id"]
                new_url = Helpers.build_tweet_url(new_id, tw_flat["author"]["username"])
                new_created_at = Helpers.twitter_to_db_time(tw_flat["created_at"])

                new_tweet = (new_id, new_url, new_created_at, old_id)

                log.debug(new_tweet)

            if found_count > 1 and self.mode == "multiproc":
                log.info(f"\nMultiple matching tweets found for {new_id}")
                log.info(f"Db idx {idx_found}")

        if found_count > 1:
            log.info(f"\nMultiple matching tweets found for {new_id}")

        return new_tweet

    def insert(self):
        pass


# TODO:
# add insert_from_file.py
# add insert_from_url.py
