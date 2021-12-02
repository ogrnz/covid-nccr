# The Impact of the COVID-19 Pandemic on Bordering Discourses Regarding Migration and Mobility in Europe

_More information about the project itself is available [here](https://nccr-onthemove.ch/projects/the-impact-of-the-covid-19-pandemic-on-bordering-discourses-regarding-migration-and-mobility-in-europe/)._

This repository contains the code for the above-mentioned project. It is structured in two main parts. First, the retrieval and storing of tweets and second, subsequent data wrangling tasks and classification.

## Setup
Each file was tested using python `3.9`. You will need to install the required packages(`requirements.txt`). Please note that `pytorch` and `scikit` are only used for classification.

## Structure
The `src` directory mainly contains files related to the first "tweets retrieval" part, whereas the `interactive` directory contains notebooks or interactive python scripts related to classification or data handling tasks.

## 1. Tweets retrieval

The goal was to setup a pipeline on a dedicated server to regularly update a database of tweets from a curated list of actors. The process' complete pipeline can be described as follows:
1. Scrape, classify(see next and upload new tweets to database (see `src/scrape.py`)
2. Export the updated database to the `database/csv` folder and convert them to xslx (see `convert_csv.py`). 

As of now, the goal is to regularly scrape tweets from a list of international actors and constitute a database. This is done with the `src/scrape.py` script.

Tweets are then classified as being about covid or not with a very basic keyword matching algorithm. See `src/common/classify.py`. The keyword list is available in `src/resources/covid_keywords.txt`.

The process' complete pipeline is the following:

1. Scrape, classify and upload new tweets (see `src/scrape.py`)
2. Export the updated database to the `database/csv` folder and convert them to xslx (see `convert_csv.py`).
3. Upload the converted files to a server via WebDav for other researchers to use.

To execute it, run `src/run.py`. 

### Remark:
- As of August 23 2021, we *no longer use the above method to scrape tweets*. Parts 2 and 3 of the pipeline remain unchanged. Instead, we use [twarc2](https://github.com/docnow/twarc/).

## 2. Classification
The database that we created is analyzed by researchers to measure the "Impact of the COVID-19 Pandemic on Bordering Discourses Regarding Migration and Mobility in Europe". In order to do that, tweets first need to be classified as being about covid or not. Then, each tweet has to be categorized given an established _codebook_ (to be made available). This was previously done manually. The second part of this project aims at automating those classification tasks.

### 2.1 Covid
The objective is to classify a given tweet as being about covid or not. For this, many different algorithms were tested. The development steps can be observed in `interactive/classifier/`. We went from a naive, keyworkds-based classifier to a majority voting classifier between 3 different algorithms. The latter is the last iteration of the classifier and is the one used. It can be used with the `interactive/classifier/ML_v2_predict.py`. Note that the pickle file of the trained model is not yet available.

### 2.2 Codebook classification
[Active research]

## Notes
- An exploratory data analysis of the database is available [here](https://github.com/ogrnz/covid-project-helpers/blob/main/interactive/descriptives/eda.ipynb).
- Many tasks handling bridging the gap between the database and Excel for the other researchers had to be automated and tested. For this, we deemed jupyter's notebooks unsuitable and preferred standalone scripts. Those are for instance the `src/insert_from_*.py`.
- `src/complete.py` is used to get existing tweets' full text
- `src/classify_all_tweets.py` is used to classify again the whole database
