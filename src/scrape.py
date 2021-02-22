import os
from datetime import date
from datetime import datetime as dt
import time
import json

import tweepy

from src.common.app import App
from src.common.helpers import Helpers
from src.db.database import Database

if __name__ == "__main__":
    
    app = App(debug=True)
    db = Database('tweets-tests.db')



