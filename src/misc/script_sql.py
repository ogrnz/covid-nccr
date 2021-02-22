'''
Get tweets from database
Complete the tweet if it is truncated
'''
import os
import sqlite3
from sqlite3 import Error
import requests
import time
from dotenv import load_dotenv
load_dotenv()

bearer_token = os.getenv('BEARER_TOKEN')
consumer_key = os.getenv('KEY')
consumer_secret = os.getenv('KEY_SECRET') 
access_key = os.getenv('TOKEN')
access_secret = os.getenv('TOKEN_SECRET')
DEBUG = False

#API
def create_url(id):
    url = "https://api.twitter.com/1.1/statuses/show/{}.json?tweet_mode=extended".format(str(id))
    return url

def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers

def connect_to_endpoint(url, headers):
    response = requests.request("GET", url, headers=headers)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()

#SQL
def connect_sqlite(db):
    conn = None
    try:
        conn = sqlite3.connect(f"../sqlite/db/{db}")
    except Error as e:
        print(e)

    return conn

def get_rts(conn):
    cur = conn.cursor()
    cur.execute("SELECT * FROM tweets WHERE type='Retweet'")

    return cur.fetchall()

def update_tweet(conn, id, text):
    sql = '''UPDATE tweets
             SET text = ?
             WHERE tweet_id = ?'''
    cur = conn.cursor()
    cur.execute(sql, (text, id,))
    conn.commit()

def counterUpdater(count, total):
    print(f"{count} / {total}", end="\r") 

def get_full_tweet(url, headers):
    json_response = connect_to_endpoint(url, headers)
    return json_response['retweeted_status']['full_text']

def main():
    conn = connect_sqlite('tweets.db')

    rts = get_rts(conn)
    tot = len(rts)
    
    headers = create_headers(bearer_token)

    errorsLst = {}
    for index, rt in enumerate(rts, start=1):
        counterUpdater(index, tot)
        url = create_url(rt[0])

        try:
            full_text = get_full_tweet(url, headers)
            time.sleep(1)
            with conn:
                update_tweet(conn, rt[0], full_text)
        except Exception as e:
            errorsLst[index] = str(e)
            print(index, rt[0], e)

if __name__ == "__main__":
    t1 = time.time()

    main()

    elapsed = time.time() - t1
    print(f'Done in {round(elapsed / 60, 2)} min')