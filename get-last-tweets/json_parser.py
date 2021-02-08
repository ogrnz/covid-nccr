import json

with open('tweets.json', 'r') as fjson:
    tweets = json.load(fjson)

outtweets = {}
for index, tweet in enumerate(tweets):
    outtweets[index] = {
        'tweet_id': tweet['id'], 
        # 'created_at': tweet.created_at, 
        'handle': tweet['user']['screen_name'],
        'type': tweet['in_reply_to_status_id'],
        'name': tweet['user']['name'],
        # 'text': tweet.text,
        'retweets': tweet['retweet_count'],
        'favorites': tweet['favorite_count'],
        'URL': f"https://twitter.com/{tweet['user']['screen_name']}/status/{tweet['id']}",
    }

    if tweet['retweeted']:
        print(f"https://twitter.com/{tweet['user']['screen_name']}/status/{tweet['id']}")
    print("----------------------------------")
    print(tweet['text'])
    print(f"https://twitter.com/{tweet['user']['screen_name']}/status/{tweet['id']}")
    print("----------------------------------")
