"""
Converts sqlite to csv
"""

import csv
from datetime import date

from common.database import Database


def convert_by_columns(
    db_name: str = "tweets_tests.db",
    cols: tuple = (
        "tweet_id",
        "covid_theme",
        "created_at",
        "handle",
        "name",
        "oldtext",
        "text",
        "url",
        "type",
        "retweets",
        "favorites",
    ),
):
    """
    Convert tweets table to csv by columns
    """

    today = str(date.today())
    db = Database(db_name)

    outfile = db_name.strip(".db")
    with db, open(
        f"database/csv/{outfile}-{today}.csv", "w+", encoding="utf-8", newline=""
    ) as out:
        try:
            writer = csv.writer(out, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(cols)
            tweets = db.get_fields(list(cols), limit=100)
            writer.writerows(tweets)
        except Exception as error:
            print("Error", error)


# Newline problem
convert_by_columns()
