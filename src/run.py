"""
Main script
1. scrape tweets
2. export database to csv and convert to xlsx
3. upload xlsx to server via WebDAV
"""

import os
import time
import logging

from common.app import App
from common.database import Database

import scrape
import convert_csv
import export_webdav

log = logging.getLogger(os.path.basename(__file__))

if __name__ == "__main__":
    t1 = time.perf_counter()

    app_run = App(debug=False)
    database = Database("tweets.db", app=app_run)

    # 1.
    log.info(
        "\n1. Scrape and classify tweets -----------------------------------------"
    )
    # Scrape and classify tweets
    # and insert them into db
    scrape.main(app_run, database)

    # 2.
    log.info("\n2. Exporting to xls -----------------------------------------")
    # Export one time total db, one time only covid
    xls_covid = convert_csv.main(database, app_run, only_covid=True)
    xls_total = convert_csv.main(database, app_run, only_covid=False)

    # 3.
    log.info(
        "\n3. Exporting to nextcloud server -----------------------------------------"
    )
    export_webdav.main(xls_covid, app=app_run)
    export_webdav.main(xls_total, app=app_run)

    elapsed = time.perf_counter() - t1
    log.info(
        f"Done. Total execution time {round(elapsed/60, 2)}min ({round(elapsed, 1)}s)."
    )
