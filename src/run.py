"""
Main script
1. scrape tweets
2. export database to csv and convert to xlsx
3. upload xlsx to server via WebDAV
"""

import time

from common.app import App
from common.database import Database

import scrape
import convert_csv
import export_webdav

if __name__ == "__main__":
    t1 = time.time()

    app_run = App(debug=False)
    database = Database("tweets.db")

    # 1.
    print("1. ----------------------------------")
    scrape.main(app_run, database)

    # 2.
    print("2. ----------------------------------")
    # By default, only export tweets classified as covid
    xls = convert_csv.main(database, app_run, only_covid=True)

    # 3.
    print("3. ----------------------------------")
    export_webdav.main(app_run, xls)

    elapsed = time.time() - t1
    print(
        f"Done. Total execution time {round(elapsed/60, 2)}min ({round(elapsed, 1)}s)"
    )
