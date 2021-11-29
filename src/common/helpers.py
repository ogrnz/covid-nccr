"""
Helpers module
"""

import os
import logging
import re
import hashlib
from datetime import datetime

import pandas as pd

log = logging.getLogger(os.path.basename(__file__))


class Helpers:
    """
    Class that contains helper functions
    """

    coded_cols = [
        "topic",
        "subcat",
        "position",
        "frame",
        "theme_hardcoded",
    ]
    topics_cov = [
        "601",
        "601.0",
        601,
        "602",
        "602.0",
        602,
        "603",
        "603.0",
        603,
        "604",
        "604.0",
        604,
        "605",
        "605.0",
        605,
        "606",
        "606.0",
        606,
        "607",
        "607.0",
        607,
    ]
    topics_not_cov = ["608", "608.0", 608]

    # !! Structure should match SQL schema. !!
    schema_cols = [
        "tweet_id",
        # "hash",
        # "retrieved_at",
        "covid_theme",
        "created_at",
        "handle",
        "name",
        "old_text",
        "text",
        "url",
        "type",
        "retweets",
        "favorites",
        "topic",
        "subcat",
        "position",
        "frame",
        "theme_hardcoded",
    ]

    twitter_time_format = "%Y-%m-%dT%H:%M:%S.000Z"
    database_time_format = "%d/%m/%Y %H:%M:%S"

    # Time range of interest
    # ]start, end[
    start = "31/12/2019"
    end = "01/04/2021"

    @staticmethod
    def sort_timerange(df: pd.DataFrame, field="date") -> pd.DataFrame:
        """
        Given a time frame, return only tweets that are in the timerange of interest.
        """

        df[field] = pd.to_datetime(df[field], format="%d/%m/%Y")
        return df[
            (df[field] > pd.to_datetime(Helpers.start, format="%d/%m/%Y"))
            & (df[field] < pd.to_datetime(Helpers.end, format="%d/%m/%Y"))
        ]

    @staticmethod
    def categorize_df_covid(df: pd.DataFrame) -> pd.DataFrame:
        """
        Return only tweets about covid given the following methodology:

        (topic in topics_cov)  -> 1.1
        OR {(covid_theme == 1)
            BUT NOT IF ((topic in topics_not_cov) OR (theme_hardcoded == 0))}  -> 1.2
        OR ((topic is None) AND (covid_theme == 1) AND (theme_hardcoded is None))  -> 1.3

        Textual explanation:
        1.1 Coded tweets (601 to 607)
        1.2 Tweets automatically classified as being about covid (covid_theme=1). From those, do not consider the ones coded as 608 or manually excluded (theme_hardcoded=0)
        1.3 Tweets about covid that are still not coded
        """
        return df[
            (df["topic"].isin(Helpers.topics_cov))
            | (
                (df["covid_theme"] == 1)
                & (
                    (~df["topic"].isin(Helpers.topics_cov)) | df["theme_hardcoded"]
                    == "0"
                )
            )
            | (
                (df["topic"].isna())
                & (df["covid_theme"] == 1)
                & (df["theme_hardcoded"].isna())
            )
        ]

    @staticmethod
    def categorize_df_not_covid(df: pd.DataFrame) -> pd.DataFrame:
        """
        Return only tweets NOT about covid given the following methodology:

        (topic == 608)  -> 2.1
        OR ((theme_hardcoded == 0) BUT NOT IF (topic in topics_cov)) -> 2.2
        OR ((covid_theme == 0)
            BUT NOT IF (topic in topics_cov))  -> 2.3

        Textual explanation:
        2.1 Tweets coded as 608
        2.2 Tweets manually excluded. From those, do not consider the ones with topic different from 608
        2.3 Tweets automatically classified as being not about covid (covid_theme=0). From those, do not consider tweets that have been coded
        """

        return df[
            ((df["covid_theme"] == 0) & (~df["topic"].isin(Helpers.topics_cov)))
            | ((df["theme_hardcoded"] == "0") & (~df["topic"].isin(Helpers.topics_cov)))
            | (df["topic"].isin(Helpers.topics_not_cov))
        ]

    @staticmethod
    def convert_date(datestr: str) -> str:
        """
        Gets a date in US format and return it EU

        2020-04-01 -> 01/04/2020
        """

        datestr = datestr.split(" ")[0]
        datestr = datestr.replace("-", "/")

        # If US format (2020/04/01), convert to EU
        datestr_split = datestr.split("/")
        if len(datestr_split[0]) == 4:
            return "/".join(datestr_split[::-1])
        return datestr

    @staticmethod
    def get_actors_url(filename: str) -> list:
        """
        Retrieve all accounts URLs located in file
        """

        with open(f"src/resources/{filename}", "r") as file:
            urls = file.readlines()
            urls = [url.strip("\n") for url in urls]
        return urls

    @staticmethod
    def dynamic_text(text: str):
        """
        Dynamically write text to the terminal
        """

        print(text, end="\r")

    @staticmethod
    def get_screen_names(urls: list) -> list:
        """
        Returns screen_names list stripped from urls
        """

        return [url[19 : len(url)] for url in urls]

    @staticmethod
    def print_timer(elapsed):
        """
        Print elapsed time in terminal
        """

        logging.info(f"\nDone in {round(elapsed, 1)} s")

    @staticmethod
    def extract_ids_file(file_path, col="URL", retdf=False) -> list:
        """
        Extract twitter statuses ids from
        a column of a xlsx (or pkl) file
        """

        if file_path[-1] == "l":
            df = pd.read_pickle(file_path)
        else:
            df = pd.read_excel(file_path)

        df[col].fillna(0, inplace=True)
        urls = df[col].values.tolist()

        ids = [Helpers.extract_id(str(url)) for url in urls]
        if retdf:
            df["tweet_id"] = df[col].apply(Helpers.extract_id)
            return (ids, df)

        return ids

    @staticmethod
    def extract_id(url: str):
        """
        Extract the status id from a twitter url
        """

        url = str(url)
        try:
            tweet_id = re.search(r"/status/(\d+)", str(url)).group(1)
        except AttributeError:
            tweet_id = 0

        return tweet_id

    @staticmethod
    def count_null_id(lst: list, start=0, finish=None):
        """
        Count the None elems in a list
        Here a null_id == 0
        """

        return sum(elem == 0 for elem in lst[start:finish])

    @staticmethod
    def count_nans(lst: list, start=0, finish=None):
        """
        Count the None elems in a list
        """

        return sum(elem is None for elem in lst[start:finish])

    @staticmethod
    def df_from_db(tweets: list) -> pd.DataFrame:
        """
        Transform list of tweets from database into pd.DataFrame
        """

        return pd.DataFrame(
            tweets,
            columns=Helpers.schema_cols,
        )

    @staticmethod
    def update_df_with_dict(df: pd.DataFrame, new_vals: dict):
        """
        Update rows of pandas DataFrame with a dict.
        """

        df.loc[
            df["tweet_id"] == new_vals["tweet_id"], new_vals.keys()
        ] = new_vals.values()

    @staticmethod
    def get_cols_as_tuple_str():
        """
        Return database's columns in string surrounded by parentheses
        """

        return str(tuple(Helpers.schema_cols))

    @staticmethod
    def get_hash(created_at, old_text, text):
        """
        Not in use yet.
        """

        # Check if created_at has the correct format

        # Concatenate the fields as strings
        to_hash = created_at + old_text + text

        # Return hash
        return hashlib.sha1(bytes(to_hash, "utf-8")).hexdigest()

    @staticmethod
    def build_tweet_url(tweet_id: str, handle: str):
        # sourcery skip: simplify-fstring-formatting
        """
        Build a valid tweet (RT, RY, New) URL with the tweet id and handle.
        https://twitter.com/{handle}/status/{id}
        """

        if handle[0] == "@":
            handle = handle[1:]
        return f"https://twitter.com/{handle}/status/{str(tweet_id)}"

    @staticmethod
    def twitter_to_db_time(date: str):
        """
        Convert a datetime string from twitter format ("%Y-%m-%dT%H:%M:%S.000Z") to
        chosen database format (Helpers.database_time_format).
        """

        tmp_date = datetime.strptime(date, Helpers.twitter_time_format)
        return datetime.strftime(tmp_date, Helpers.database_time_format)

    @staticmethod
    def get_type_from_json(line: str):
        """
        Detects if a given tweet (jsonl format) is Retweet, Reply or New.
        """

        if line["text"][:2] == "RT":  # Is RT
            # if "retweeted_status" in line:  # Not in jsonl...
            tweet_type = "Retweet"
        else:  # Not a Retweet
            tweet_type = "New"
            if "in_reply_to_user_id" in line:
                tweet_type = "Reply"

        return tweet_type


if __name__ == "__main__":
    Helpers.extract_ids_file("src/resources/data/fr.pkl")
