import tweepy
import csv
from datetime import date
from datetime import datetime as dt
import time
import os
import pandas as pd
import numpy as np
import json
from dotenv import load_dotenv
load_dotenv()

BEARER_TOKEN = os.getenv('BEARER_TOKEN')
consumer_key = os.getenv('KEY')
consumer_secret = os.getenv('KEY_SECRET') 
access_key = os.getenv('TOKEN')
access_secret = os.getenv('TOKEN_SECRET')
DEBUG = True

def get_actors_urls(filename):
    with open(filename, 'r') as f:
        urls = f.readlines()
    urls = [url.strip('\n') for url in urls]
    return urls

def setup_API(consumer_key, consumer_secret, access_key, access_secret):
    try:
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_key, access_secret)
        api = tweepy.API(auth)
        return api

    except Exception as e:
        return e

def get_all_tweets(screen_name, api):
    '''
    Script from @yanofsky as baseline
    https://gist.github.com/yanofsky/5436496
    '''

    alltweets = []
    new_tweets = api.user_timeline(screen_name=screen_name, count=200)
    alltweets.extend(new_tweets)

    oldest = alltweets[-1].id - 1

    while len(new_tweets) > 0:
        print(f"getting tweets before {oldest}")

        new_tweets = api.user_timeline(
            screen_name=screen_name, count=200, max_id=oldest)

        alltweets.extend(new_tweets)
        oldest = alltweets[-1].id - 1
        
        print(f"...{len(alltweets)} tweets downloaded so far")
        if DEBUG:
            break
    
    #Convert to dict
    outtweets = {}
    for index, tweet in enumerate(alltweets):

        #Ignore tweets older than 2019/12/31
        as_of = dt.strptime("2019/12/31", "%Y/%m/%d")
        if tweet.created_at < as_of:
            pass  

        tweet_type = 'Reply'
        if tweet.in_reply_to_status_id is None:
            tweet_type = 'New'
        elif tweet.text[:2] == 'RT':
            tweet_type = 'Retweet'

        outtweets[index] = {
            'tweet_id': tweet.id_str, 
            'type': tweet_type,
            'created_at': tweet.created_at, 
            'handle': tweet.user.screen_name,
            'name': tweet.user.name,
            'text': tweet.text,
            'URL': f'https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}',
            'retweets': tweet.retweet_count,
            'favorites': tweet.favorite_count
        }

    return outtweets

if __name__ == "__main__":
    today = date.today()
    date_name = today.strftime('%Y-%m-%d')

    # Get actors urls
    urls = get_actors_urls('actors_url.txt')
    screen_names = [url[19:len(url)] for url in urls]

    # Connect to Twitter API
    api = setup_API(consumer_key, consumer_secret, access_key, access_secret) 

    t1 = time.time()
    total = len(screen_names)

    total_tweets = {}
    for i, name in enumerate(screen_names):
        print(i + 1, '/', total)
        print('Starting to retrieve tweets for ' + name)

        try:
            # Get last ~3200 tweets from someone
            tweets = get_all_tweets(name, api)
        except Exception as e:
            print('ERROR', name, e)

        # Add to total dict
        total_tweets[i] = tweets

        if DEBUG:
            break

    elapsed = time.time() - t1
    print(f'Done in {round(elapsed / 60, 2)} min')