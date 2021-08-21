import sys, os

sys.path.append(os.path.abspath("src"))

import time
import cProfile
import pstats

from common.app import App
from common.database import Database
from common.convert import Converter


def profiling(file):
    with cProfile.Profile() as profile:
        converter.csv_to_xlsx_pyexcel(file)

    stats = pstats.Stats(profile)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    stats.dump_stats(filename="convert_csv.prof")


if __name__ == "__main__":
    app_run = App(debug=False)
    db = Database("tweets.db", app=app_run)

    converter = Converter(database=db, only_covid=False, app=app_run)

    print("Converting table to csv...")
    csv_file = converter.convert_by_columns()

    print("Converting csv to xlsx...")
    # profiling(csv_file)

    funcs = [
        converter.csv_to_xlsx,
        converter.csv_to_xlsx_v2,
        converter.csv_to_xlsx_pd,
        converter.csv_to_xlsx_pd_v2,
        converter.csv_to_xlsx_pyexcel,
    ]

    for i, func in enumerate(funcs):
        start = time.time()
        print(i)
        func(csv_file)
        print(i, time.time() - start)
        time.sleep(2)
