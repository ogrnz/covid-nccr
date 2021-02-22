"""
App module
"""

import os

from dotenv import load_dotenv

class App:
    """
    Main App class
    """
    
    BEARER_TOKEN = None
    CONSUMER_KEY = None
    CONSUMER_SECRET = None
    ACCESS_KEY = None
    ACCESS_SECRET = None

    DEBUG = False

    def __init__(self, debug: bool):
        load_dotenv(dotenv_path='.env')

        self.BEARER_TOKEN = os.getenv('BEARER_TOKEN')
        self.CONSUMER_KEY = os.getenv('KEY')
        self.CONSUMER_SECRET = os.getenv('KEY_SECRET')
        self.ACCESS_KEY = os.getenv('TOKEN')
        self.ACCESS_SECRET = os.getenv('TOKEN_SECRET')
        self.DEBUG = debug
