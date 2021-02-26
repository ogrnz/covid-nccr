# IT part

### The Impact of the COVID-19 Pandemic on Bordering Discourses Regarding Migration and Mobility in Europe

_More information about the project itself is available [here](https://nccr-onthemove.ch/projects/the-impact-of-the-covid-19-pandemic-on-bordering-discourses-regarding-migration-and-mobility-in-europe/)._

As of now, the goal is to regularly scrape tweets from a list of international actors and constitute a database. This is done with the `src/scrape.py` script.

Tweets are then classified as being about covid or not with a very basic keyword matching algorithm. See `src/common/classify.py`. The keyword list is available in `src/resources/covid_keywords.txt`.

The process' complete pipeline is the following:

1. Scrape, classify and upload new tweets (see `src/scrape.py`)
2. Export the updated database to the `database/csv` folder and convert them to xslx (see `convert_csv.py`).

To execute it, run `src/run.py` with the installed required packages (`requirements.txt`).

### Misc
- `src/complete.py` is used to get existing tweets' full text
- `src/classify_all_tweets.py` is used to classify again the whole database

TODO:

-   logging
-   set up server
-   better classifier
-   ...
