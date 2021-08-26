"""
Import jsonl FLATTENED file into the database.
Already present tweets are updated.
"""

import time
import json

from common.app import App
from common.api import Api
from common.database import Database
from common.helpers import Helpers
from common.classify import Classifier
from common.insertor import InsertFromJsonl

def main():
    test_file = "UDCch_flat.jsonl"
    app = App()
    api = Api(app)
    insertor = InsertFromJsonl(app)

    jsonl = insertor.read(test_file)
    
    for tweet in jsonl:
        
        break

if __name__ == "__main__":
    main()