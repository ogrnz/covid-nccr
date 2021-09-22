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
topics_cov = [
    "601",
    "601.0",
    "602",
    "602.0",
    "603",
    "603.0",
    "604",
    "604.0",
    "605",
    "605.0",
    "606",
    "606.0",
    "607",
    "607.0",
]
topics_not_cov = ["608", "608.0"]


# All coded tweets
all_coded = df[
    (df["topic"].isin(topics_cov + topics_not_cov)) | (df["theme_hardcoded"] == "0")
]
all_yes = all_coded[all_coded["topic"].isin(topics_cov)]
all_no = all_coded[
    all_coded["topic"].isin(topics_not_cov) | (all_coded["theme_hardcoded"] == "0")
]

# %%
# Calculate false negative
false_neg_count = sum(all_yes["covid_theme"] == "0")
all_count = len(all_yes)

print("False Negative:")
print(
    f"Out of {all_count} manually coded tweets about covid, {false_neg_count} were classified as not being about covid although they were. The false negative rate is {round(false_neg_count / all_count*100, 1)}%."
)
# 0%, but it's on the train data, so it's quite normal

#%%
# Calculate false positive

false_pos_count = sum(all_no["covid_theme"] == 1)
all_excluded_count = len(all_no)

print("False Positive:")
print(
    f"Out of {all_excluded_count} manually classified tweets, {false_pos_count} were classified as being about covid although they were not. The false positive rate is {round(false_pos_count / all_excluded_count*100, 1)}%."
)
# 0.1% but "real" is 1.8%

# %%
