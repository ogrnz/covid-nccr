import pandas as pd
import json
import requests
import re
import time
import os
from dotenv import load_dotenv
load_dotenv()

BEARER_TOKEN = os.getenv('BEARER_TOKEN')
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

#Helpers
def get_id(url):
    tweet_id = re.search(r'/status/(\d+)', url).group(1)
    return tweet_id

def get_full_tweet(url, headers):
    json_response = connect_to_endpoint(url, headers)
    return json_response['retweeted_status']['full_text']

def counterUpdater(count, total):
    print(f"{count} / {total}", end="\r") 

def main(df, delay=1):
    headers = create_headers(BEARER_TOKEN)
    shape1 = df.shape
    errorsLst = {}
    total = len(df)
    t1 = time.time()
    count = 0

    for index, row in df.loc[1100:1500,:].iterrows():
        counterUpdater(index, total)

        if row['Text'][-1] == '…':
            count += 1
            tweet_id = get_id(row['URL'])
            url = create_url(tweet_id)
            try:
                time.sleep(delay)
                fullText = get_full_tweet(url, headers)
                df.loc[index, 'Text'] = fullText
            except Exception as e:
                #might be better to log directly
                errorsLst[index] = str(e)
                print(index, e)
                pass
            except KeyboardInterrupt:
                return df, errorsLst 

        if DEBUG==True and index==3:
            break

    shape2 = df.shape
    if shape1 != shape2:
        print(f"ERROR: {df} shapes differ")

    elapsed = time.time() - t1
    print(f'Edited {count} tweets')
    print(f'Done in {round(elapsed / 60, 2)} min [1s delay per request]')

    return df, errorsLst   

if __name__ == "__main__":
    #eu = pd.read_excel(r'data/EU.xlsx')
    #eu.to_pickle(r'data/EU.pkl')
    #un = pd.read_excel(r'data/UN.xlsx')
    #un.to_pickle(r'data/UN.pkl')
    #eu = pd.read_pickle(r'data/EU.pkl')
    un = pd.read_pickle(r'data/UN.pkl')

    #eu = eu.set_index('ID')
    un = un.set_index('ID')

    dfs = [un]
    for i, df in enumerate(dfs):
        df['OldText'] = df['Text'] 
        name = 'EU' if df.shape[0] == 7595 else 'UN'

        #insert OldText before Text
        cols = df.columns.tolist()
        cols.insert(6, cols[-1])
        cols.pop()
        df = df[cols]

        print("Starting...")

        try:
            edited, errorsLst = main(df)
            print('Writing to xlsx.')   
            edited.to_excel('data/' + name + '_total.xlsx')
            if len(errorsLst) > 0:
                with open(f'logs/{name}_errors.json', 'w') as outfile:
                    json.dump(errorsLst, outfile)
            print('Done.')
        except Exception as e:
            print('Exception', e)
        
# rate limit 900/15min -> 1 tweet/s (v1.1) 