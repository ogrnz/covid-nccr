"""
App module
"""

import os
import logging

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

    root_dir = None

    debug = False

    def __init__(self, debug: bool = False, root_dir: str = ""):
        self.debug = debug

        abspath = os.path.abspath(__file__)
        dirname = os.path.dirname
        root_dir = dirname(dirname(dirname(abspath)))
        os.chdir(root_dir)
        self.root_dir = root_dir

        # Setup logging
        logging_level = logging.INFO
        if self.debug:
            logging_level = logging.DEBUG

        logging.basicConfig(
            level=logging_level,
            filename="app.log",
            format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
        )

        load_dotenv(load_dotenv(dotenv_path=f"{root_dir}/.env"))

        self.bearer_token = os.getenv("BEARER_TOKEN")
        self.consumer_key = os.getenv("KEY")
        self.consumer_secret = os.getenv("KEY_SECRET")
        self.access_key = os.getenv("TOKEN")
        self.access_secret = os.getenv("TOKEN_SECRET")

        self.webdav_server = os.getenv("WEBDAV_SERVER")
        self.webdav_login = os.getenv("WEBDAV_LOGIN")
        self.webdav_pwd = os.getenv("WEBDAV_PWD")
