# pylint: skip-file
#%%
import sys, os

sys.path.append(os.path.abspath(os.path.join("..", "src")))

#%%
import pandas as pd

from common.database import Database
from common.app import App
from common.helpers import Helpers
from common.api import Api

#%%
app_run = App(debug=False)
db = Database("tweets.db", app=app_run)

# %%
from common.classify import Classifier

classifier = Classifier()

#%%
with db:
    print("Downloading from db...")
    tweets = db.get_fields(["tweet_id", "covid_theme", "text"])
len(tweets)

#%%
def classify(istr):
    if not classifier.classify(istr):
        return 0
    return 1


#%%
df = pd.DataFrame(tweets)
df = df.rename(columns={0: "id", 1: "theme", 2: "text"})
df["theme"] = 1
df["theme"] = df["text"].apply(classify)

#%%
df["theme"].unique()

#%%
new_tweets = df.values.tolist()
classified_tw = []

for new_tweet in new_tweets:
    tw = (new_tweet[1], new_tweet[0])
    classified_tw.append(tw)

#%%
classified_tw[123230]

#%%
with db:
    inserted = db.update_theme_many(classified_tw)
print(f"{inserted} tweets inserted")
