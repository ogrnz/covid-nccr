"""
Converter module
"""

import csv
import logging
import os
import os.path
import uuid
from datetime import date

import xlsxwriter

from common.app import App
from common.database import Database
from common.helpers import Helpers

log = logging.getLogger(os.path.basename(__file__))


class Converter:
    """
    Convert database table to csv
    """

    def __init__(self, app: App, database: Database, only_covid: bool):
        self.app = app
        self.database = database
        self.only_covid = only_covid

    def __file_exists(self, file_name):
        """
        Check if a file of that name already exists in the database/xlsx folder
        """

        return bool(
            os.path.isfile(f"{self.app.root_dir}/database/xlsx/{file_name}.xlsx")
        )

    def __dir_exists(self, dirname: str) -> bool:
        """
        Check if directory exists in database folder
        """

        return bool(os.path.isdir(f"{self.app.root_dir}/database/{dirname}"))

    def __gen_csv_reader(self, file):
        with open(
            f"{self.app.root_dir}/database/csv/{file}", encoding="utf8", newline=""
        ) as open_f:
            reader = csv.reader(open_f)
            yield from reader

    def convert_by_columns(self, cols=tuple(Helpers.schema_cols)):
        """
        Convert tweets table to csv by columns
        """

        # Check if database/csv/ dir exists
        if not self.__dir_exists("csv"):
            os.mkdir(f"{self.app.root_dir}/database/csv/")

        today = str(date.today())

        covid = "Covid-" if self.only_covid else "Tot-"
        outfile = self.database.db_name
        outfile = outfile.strip(".db")

        with self.database, open(
            f"{self.app.root_dir}/database/csv/{covid}{outfile}-{today}.csv",
            "w+",
            encoding="utf-8",
            newline="",
        ) as out:
            try:
                writer = csv.writer(out, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
                writer.writerow(cols)
                tweets = self.database.get_fields(list(cols), self.only_covid)
                writer.writerows(tweets)
                log.info(
                    f"Table successfully converted as database/csv/{covid}{outfile}-{today}.csv"
                )

                return f"{covid}{outfile}-{today}.csv"
            except csv.Error as error:
                log.warning(f"Error while writing CSV {error}")

    def csv_to_xlsx(self, csv_file):
        """
        Export a csv file to xlsx
        """

        # Check if database/xlsx/ dir exists
        if not self.__dir_exists("xlsx"):
            os.mkdir(f"{self.app.root_dir}/database/xlsx/")

        csv_file = csv_file.strip(".csv")
        orig_name = csv_file
        if self.__file_exists(csv_file):
            csv_file = orig_name + "-" + str(uuid.uuid4())[:7]
            if self.__file_exists(csv_file):
                csv_file = orig_name + "-" + str(uuid.uuid4())[:7]

        workbook = xlsxwriter.Workbook(
            f"{self.app.root_dir}/database/xlsx/{csv_file}.xlsx",
            {"constant_memory": True, "strings_to_urls": False},
        )
        wk_sheet = workbook.add_worksheet()

        with open(
            f"{self.app.root_dir}/database/csv/{orig_name}.csv",
            "r",
            encoding="utf8",
            newline="",
        ) as file:
            reader = csv.reader(file)
            for row_i, row in enumerate(reader):
                for col_i, val in enumerate(row):
                    wk_sheet.write(row_i, col_i, val)

        workbook.close()
        log.info(f"File successfully exported to database/xlsx/{csv_file}.xlsx")

        return f"{csv_file}.xlsx"
