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
        converter.csv_to_xlsx(file)

    stats = pstats.Stats(profile)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    stats.dump_stats(filename="convert_csv.prof")


if __name__ == "__main__":
    app_run = App(debug=False)
    db = Database("tweets.db", app=app_run)

    converter = Converter(database=db, only_covid=False, app=app_run)

    funcs = [converter.csv_to_xlsx_v4]

    for i, func in enumerate(funcs):
        start = time.time()
        print(i)
        func("Tot-tweets-2021-08-21.csv")
        print(i, time.time() - start)
        time.sleep(2)
