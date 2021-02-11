import tweepy

from datetime import date
from datetime import datetime as dt
import time

import os

import pandas as pd
import numpy as np
import json

import sqlite3
from sqlite3 import Error

from dotenv import load_dotenv
load_dotenv()

BEARER_TOKEN = os.getenv('BEARER_TOKEN')
consumer_key = os.getenv('KEY')
consumer_secret = os.getenv('KEY_SECRET') 
access_key = os.getenv('TOKEN')
access_secret = os.getenv('TOKEN_SECRET')
DEBUG = False

# Sqlite
def connect_sqlite(db):
    conn = None
    try:
        conn = sqlite3.connect(f"../sqlite/db/{db}")
    except Error as e:
        print(e)

    return conn
def create_table(conn, table_sql):
    try:
        c = conn.cursor()
        c.execute(table_sql)
    except Error as e:
        print(e)
def insert_tweet(conn, tweet):
    sql = ''' INSERT OR IGNORE INTO tweets(
                    tweet_id,
                    covid_theme,
                    type, 
                    created_at, 
                    handle, 
                    name, 
                    oldtext,
                    text,
                    url,
                    retweets,
                    favorites)
              VALUES(?,?,?,?,?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, tweet)
    conn.commit()

    return cur.lastrowid
def get_last_id(conn, screen_name):
    screen_name = f'@{screen_name}'

    cur = conn.cursor()
    cur.execute('''
        SELECT tweet_id 
        FROM tweets 
        WHERE handle=?
        ORDER BY tweet_id DESC''', 
        (screen_name,)
    )

    # Return last inserted tweet_id by handle
    return cur.fetchone()

# Helpers
def get_actors_urls(filename):
    with open(filename, 'r') as f:
        urls = f.readlines()
    urls = [url.strip('\n') for url in urls]
    return urls
def counterUpdater(count, total):
    print(f"{count} / {total}", end="\r") 

# Tweepy
def setup_API(consumer_key, consumer_secret, access_key, access_secret):
    try:
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_key, access_secret)
        api = tweepy.API(auth)
        return api

    except Exception as e:
        return e

def get_all_tweets(screen_name, last_id, api):
    '''
    Script from @yanofsky as baseline
    https://gist.github.com/yanofsky/5436496
    '''

    alltweets = []
    new_tweets = api.user_timeline(
        screen_name=screen_name, count=200, 
        tweet_mode="extended"
    )
    
    alltweets.extend(new_tweets)

    oldest = alltweets[-1].id - 1


    while len(new_tweets) > 0:
        # If tweet older than that ID (== 31/12/2019)
        # or older than last ID in db for that actor, go to next actor
        if oldest < 1211913001147740161 or oldest < last_id:
            break

        print(f"Getting tweets before {oldest}")

        new_tweets = api.user_timeline(
            screen_name=screen_name, 
            count=200, max_id=oldest,
            tweet_mode="extended"
        )

        alltweets.extend(new_tweets)
        oldest = alltweets[-1].id - 1
        
        print(f"...{len(alltweets)} tweets downloaded")

        if DEBUG:
            break
    
    outtweets = {}
    for index, tweet in enumerate(alltweets):

        # Ignore tweets older than 2019/12/31
        as_of = dt.strptime("2019/12/31", "%Y/%m/%d")
        if tweet.created_at < as_of:
            continue  
        
        # Get type
        # tweet_type = None
        # if tweet.in_reply_to_status_id is not None:
        #     tweet_type = 'Reply'
        # elif tweet.text[:2] == 'RT':
        #     tweet_type = 'Retweet'
        # else:
        #     tweet_type = 'New'
        # print(tweet._json)

        old_text = None
        if hasattr(tweet, 'retweeted_status'):
            old_text = tweet.full_text
            tweet_type = 'Retweet'
            full_text = tweet.retweeted_status.full_text
        else:  # Not a Retweet
            full_text = tweet.full_text
            tweet_type = 'New'
            if tweet.in_reply_to_status_id is not None:
                tweet_type = 'Reply'

        # Make final dict
        outtweets[index] = {
            'tweet_id': tweet.id,
            'covid_theme': None, 
            'type': tweet_type,
            'created_at': tweet.created_at.strftime('%d/%m/%Y %H:%M:%S'), 
            'handle': f"@{tweet.user.screen_name}",
            'name': tweet.user.name,
            'oldText': old_text,
            'text': full_text,
            'URL': f'https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}',
            'retweets': tweet.retweet_count,
            'favorites': tweet.favorite_count
        }

    #print(outtweets[0]) #RT
    #print(outtweets[1]) #Reply
    #print(outtweets[4]) #New
    return outtweets

if __name__ == "__main__":
    today = date.today()
    date_name = today.strftime('%Y-%m-%d')

    conn = connect_sqlite('tweets.db')

    # Get actors urls
    urls = get_actors_urls('actors_url.txt')
    screen_names = [url[19:len(url)] for url in urls]

    # Get actors last scraped tweet_id
    last_ids = {}
    with conn:
        for name in screen_names:
            try:
                last_ids[name] = int(get_last_id(conn, name)[0])    
            except TypeError as e:
                last_ids[name] = 0
                print('TypeError', name, e)

    # Connect to Twitter API
    api = setup_API(consumer_key, consumer_secret, access_key, access_secret) 

    t1 = time.time()
    total = len(screen_names)

    total_tweets = {}
    scrape_errors = {}

    for i, actor in enumerate(screen_names, start=1):
        print(i, '/', total)
        print('Starting to retrieve tweets for ', actor)
        
        last_tweet_id = last_ids[actor]
        try:
            # Get last ~3200 tweets from someone
            tweets = get_all_tweets(actor, last_tweet_id, api)
            # Add to total dict
            total_tweets[actor] = tweets
        except Exception as e:
            scrape_errors[actor] = str(e)
            print('ERROR', actor, e)

        if DEBUG:
            break

    elapsed = time.time() - t1
    print(f'Done in {round(elapsed / 60, 2)} min') 
    if len(scrape_errors) > 0:
        print('With some errors:', json.dumps(scrape_errors, indent=4))

    # Insert tweets into DB
    t1 = time.time()
    db_errors = {}
    
    for actor in total_tweets:
        print('Inserting tweets from', actor)

        # Open and close conn for each actor, so changes are not lost if issue  
        with conn:
            for i, tweet in enumerate(total_tweets[actor], start=1):
                counterUpdater(i, len(total_tweets[actor]))

                tweet_entry = ()
                for key, val in total_tweets[actor][tweet].items():
                    tweet_entry += (val,)

                tmp_tweet = list(tweet_entry)
                # Work with entries
                #Check if is about covid
                tmp_tweet[1] = 0
                tweet_entry = tuple(tmp_tweet)

                try:
                    insert_tweet(conn, tweet_entry)   
                except Exception as e:
                    db_errors[actor][tweet] = str(e)
                    print('ERROR', actor, e)

    elapsed = time.time() - t1
    print(f'Done in {round(elapsed / 60, 2)} min')
    if len(db_errors) > 0:
        print('With some errors:', json.dumps(db_errors, indent=4))

# TODO
# Retrieve full text with extended mode