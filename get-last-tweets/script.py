import tweepy
import csv
from datetime import date
import time
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()

BEARER_TOKEN = os.getenv('BEARER_TOKEN')
consumer_key = os.getenv('KEY')
consumer_secret = os.getenv('KEY_SECRET') 
access_key = os.getenv('TOKEN')
access_secret = os.getenv('TOKEN_SECRET')
DEBUG = False

today = date.today()
date_name = today.strftime('%Y-%m-%d')

def get_all_tweets(screen_name):
    '''
    Script from @yanofsky as baseline
    https://gist.github.com/yanofsky/5436496
    '''
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

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

    #Convert to dict for easier use with pandas
    outtweets = {}
    for index, tweet in enumerate(alltweets):
        outtweets[index] = {
            'tweet_id': tweet.id_str, 
            'created_at': tweet.created_at, 
            'text': tweet.text
        }

    df = pd.DataFrame.from_dict(outtweets).T
    df['created_at'] = pd.to_datetime(df.created_at)

    #Delete tweets older than 2020/08/31
    as_of = pd.to_datetime('2020/08/31')
    df = df[df['created_at'] > as_of]

    # with open(f'tweets-data/{screen_name}/{date_name}-{screen_name}_tweets.csv', 'w', newline='', encoding='utf-8') as f:
    #     writer = csv.writer(f, delimiter=';')
    #     writer.writerow(["id", "created_at", "text"])
    #     writer.writerows(outtweets)
    
    print('Writing to csv')
    df.to_csv(f'tweets-data/{screen_name}/{date_name}-{screen_name}.csv', index=False)
    pass

with open('actors_url.txt', 'r') as f:
    urls = f.readlines()

urls = [url.strip('\n') for url in urls]
screen_names = [url[19:len(url)] for url in urls]

t1 = time.time()
total = len(screen_names)
# Starting from GUENGL to debug
for i, name in enumerate(screen_names[12:], start=1):
    print(i, '/', total)
    print('Starting to retrieve tweets for ' + name)

    try:
        get_all_tweets(name)
    except Exception as e:
        print(name, e)

elapsed = time.time() - t1
print(f'Done in {round(elapsed / 60, 2)} min')
