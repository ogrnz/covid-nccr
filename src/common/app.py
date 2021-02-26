"""
App module
"""

import os

from dotenv import load_dotenv


class App:
    """
    Main App class
    """

    bearer_token = None
    consumer_key = None
    consumer_secret = None
    access_key = None
    access_secret = None

    debug = False

    def __init__(self, debug: bool = False):
        load_dotenv(dotenv_path=".env")

        self.bearer_token = os.getenv("BEARER_TOKEN")
        self.consumer_key = os.getenv("KEY")
        self.consumer_secret = os.getenv("KEY_SECRET")
        self.access_key = os.getenv("TOKEN")
        self.access_secret = os.getenv("TOKEN_SECRET")
        self.debug = debug
