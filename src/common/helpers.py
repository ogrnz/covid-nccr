"""
Helpers module
"""

import re

import pandas as pd


class Helpers:
    """
    Class that contains helper functions
    """

    def __init__(self):
        pass

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
        print(f"\nDone in {round(elapsed, 1)} s")

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

        # ids = []
        # for url in urls:
        #     tweet_id = Helpers.extract_id(str(url))
        #     ids.append(tweet_id)
        ids = [Helpers.extract_id(str(url)) for url in urls]

        if retdf:
            return (ids, df)
        return ids

    @staticmethod
    def extract_id(url: str):
        """
        Extract the status id from a twitter url
        """

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


if __name__ == "__main__":
    Helpers.extract_ids_file("src/resources/data/fr.pkl")
