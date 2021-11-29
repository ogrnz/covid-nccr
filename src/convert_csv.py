"""
Converts sqlite to csv and exports it to xslx
"""

import os
import logging

from common.app import App
from common.database import Database
from common.convert import Converter

log = logging.getLogger(os.path.basename(__file__))


def main(database: Database, app: App = None, only_covid=False):
    """
    Main script
    """

    converter = Converter(database=database, only_covid=only_covid, app=app)

    log.info("Converting table to csv...")
    csv_file = converter.convert_by_columns()

    log.info("Converting csv to xlsx...")
    xls = converter.csv_to_xlsx(csv_file)

    return xls


if __name__ == "__main__":
    app_run = App(debug=False)
    db = Database("tweets.db", app=app_run)

    main(db, app=app_run)
    main(db, app=app_run, only_covid=True)
