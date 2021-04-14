# pylint: skip-file
#%%
import sys, os

sys.path.append(os.path.abspath(os.path.join("..", "src")))

#%%
import pandas as pd

from common.database import Database
from common.app import App

app_run = App(debug=True)
db = Database("tweets.db", app=app_run)

#%%
with db:
    tweets = db.get_all_tweets()
print(len(tweets))
print(tweets[0])

df = pd.DataFrame(
    tweets,
    columns=[
        "tweet_id",
        "covid_theme",
        "created_at",
        "handle",
        "name",
        "oldText",
        "text",
        "URL",
        "type",
        "retweets",
        "favorites",
        "topic",
        "subcat",
        "position",
        "frame",
        "theme_hardcoded",
    ],
)
df

"""
In our setting, a false positive (falsely classified as about covid) is less harmful than a covid tweet being classified as not being one (false negative).
"""


#%%
topics = list(range(601, 608))
topics = [str(topic) for topic in topics]

# All coded tweets except coded as 608
all_coded = df[df["topic"].isin(topics)]

# %%
# Calculate false negative
false_neg_count = sum(all_coded["covid_theme"] == 0)
all_count = len(all_coded["covid_theme"])

print("False Negative:")
print(
    f"Out of {all_count} manually coded tweets, {false_neg_count} were classified as not being about covid although they were. The false negative rate is {round(false_neg_count / all_count*100, 1)}%."
)

#%%
# Calculate false positive

sub_hardcoded = df[df["theme_hardcoded"] == "0"]
all_608 = df[df["topic"] == "608"]
all_excluded = pd.concat([sub_hardcoded, all_608])

false_pos_count = sum(all_excluded["covid_theme"] == 1)

print("False Positive:")
print(
    f"Out of {len(all_excluded)} manually classified tweets, {false_pos_count} were classified as being about covid although they were not. The false positive rate is {round(false_pos_count / len(all_excluded)*100, 1)}%."
)

# %%
"""
Satisfactory false negative rate (4.2%). Would be nice to decrease the false positive as well.
"""
