"""
Converts sqlite to csv and exports it to xslx
"""

from common.app import App
from common.database import Database
from common.convert import Converter


def main(app: App, database: Database):
    converter = Converter(database, "convert")

    print("Converting table to csv...")
    csv_file = converter.convert_by_columns()

    print("Converting csv to xlsx...")
    converter.csv_to_xlsx(csv_file)


if __name__ == "__main__":
    app_run = App(debug=False)
    database = Database("tweets.db")

    main(app_run, database)
