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
    database = Database("tweets.db", app=app_run)

    # 1.
    print("\n1. Scrape and classify tweets ", "-" * 60)
    # Scrape and classify tweets
    # and insert them into db
    scrape.main(app_run, database)

    # 2.
    print("\n2. Exporting to xls ", "-" * 60)
    # Export one time total db, one time only covid
    xls_covid = convert_csv.main(database, app_run, only_covid=True)
    xls_total = convert_csv.main(database, app_run, only_covid=False)

    # 3.
    print("\n3. Exporting to nextcloud server ", "-" * 60)
    export_webdav.main(xls_covid, app=app_run)
    export_webdav.main(xls_total, app=app_run)

    elapsed = time.time() - t1
    print(
        f"Done. Total execution time {round(elapsed/60, 2)}min ({round(elapsed, 1)}s)."
    )
