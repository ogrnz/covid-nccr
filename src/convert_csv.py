"""
Converts sqlite to csv and exports it to xslx
"""

from common.app import App
from common.database import Database
from common.convert import Converter


def main(database: Database, app: App = None, only_covid=True):
    """
    Main script
    """

    converter = Converter(database, only_covid)

    print("Converting table to csv...")
    csv_file = converter.convert_by_columns()

    print("Converting csv to xlsx...")
    xls = converter.csv_to_xlsx(csv_file)

    return xls


if __name__ == "__main__":
    db = Database("tweets.db")

    print(main(db))
