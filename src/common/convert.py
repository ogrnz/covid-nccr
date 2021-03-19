"""
Converter module
"""

import csv
import uuid
import os.path
from datetime import date

import openpyxl

from common.app import App
from common.database import Database


class Converter:
    """
    Convert database table to csv
    """

    def __init__(self, app: App, database: Database, only_covid: bool):
        self.app = app
        self.database = database
        self.only_covid = only_covid

    def convert_by_columns(
        self,
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
            "topic",
            "subcat",
            "position",
            "frame",
            "theme_hardcoded",
        ),
    ):
        """
        Convert tweets table to csv by columns
        """

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
                print(
                    f"Table successfully converted as database/csv/{covid}{outfile}-{today}.csv"
                )

                return f"{covid}{outfile}-{today}.csv"
            except csv.Error as error:
                print("Error while writing CSV", error)

    def csv_to_xlsx(self, csv_file):
        """
        Export a csv file to xlsx
        """

        workbook = openpyxl.Workbook()
        sheet = workbook.active

        with open(
            f"{self.app.root_dir}/database/csv/{csv_file}",
            "r",
            encoding="utf8",
            newline="",
        ) as file:
            reader = csv.reader(file)
            for row_i, row in enumerate(reader, start=1):
                for col_i, val in enumerate(row, start=1):
                    sheet.cell(row=row_i, column=col_i).value = val
        try:
            csv_file = csv_file.strip(".csv")
            orig_name = csv_file
            if self.__file_exists(csv_file):
                csv_file = orig_name + "-" + str(uuid.uuid4())[:7]
                if self.__file_exists(csv_file):
                    csv_file = orig_name + "-" + str(uuid.uuid4())[:7]

            workbook.save(f"{self.app.root_dir}/database/xlsx/{csv_file}.xlsx")
            print(f"File successfully exported to database/xlsx/{csv_file}.xlsx")

            return f"{csv_file}.xlsx"
        except openpyxl.utils.exceptions.InvalidFileException as error:
            print("Error", error)

    def __file_exists(self, file_name):
        """
        Check if a file of that name already exists in the databvase/xlsx folder
        """

        return bool(
            os.path.isfile(f"{self.app.root_dir}/database/xlsx/{file_name}.xlsx")
        )
