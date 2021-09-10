# pylint: skip-file

"""
Still tweets with NULL as url.
value_counts:
@WHO             2222
@DrTedros         331
@MinSoliSante     267
@UN                17
@enmarchefr         1
TOT              2838
"""

#%%
%load_ext autoreload
%autoreload 2

#%%
import sys, os

# sys.path.append(os.pathpython3.abspath(os.path.join("src")))
sys.path.append(os.path.abspath(os.path.join("..", "src")))

import time
import json
import unicodedata
import multiprocessing

import pandas as pd
import numpy as np
import tqdm
from tqdm.contrib.concurrent import process_map

from common.app import App
from common.api import Api
from common.database import Database
from common.helpers import Helpers
from common.insertor import InsertFromJsonl

app_run = App(debug=False)
db = Database("tweets.db", app=app_run)

#%%
with db:
    tws_probs_all = db.get_all_tweets(("url", None))
df = Helpers.df_from_db(tws_probs_all)

#%%
# Problematic tweets from
# ['@WHO', '(@MinSoliSante)', '@DrTedros', '@UN', '@enmarchef']
# @MinSoliSante account doesn't exist anymore, won't be possible to retrieve real ids
# @MinSoliSante ok
# @DrTedros ok
# @UN ok
# @WHO ok
# @enmarchefr ok

jsonl_path = os.path.join(app_run.root_dir, "database", "jsonl")
test_file = "UN_flat_test.jsonl"
# test_file = "DrTedros_flat.jsonl"
# test_file = "UN_flat.jsonl"
# test_file = "enmarchefr_flat.jsonl"
jsonl_file_flat = os.path.join(jsonl_path, "flat", test_file)

with open(jsonl_file_flat) as jsonl_flat:
    tws_flat = [json.loads(line) for line in jsonl_flat]

#%%
# When testing, insert a fake tweet that we are sure is in the database
# so we, at least, have one match.
fake_tw = (
    "123",
    0,
    "00/00/0000",
    "@UN",
    "United Nations",
    tws_flat[0]["text"],
    tws_flat[0]["text"],
    "0",
    "New",
    None,
    None,
    None,
    None,
    None,
    None,
    "0",
)
# tws_probs_all.insert(0, fake_tw)

#%%
if __name__ == "__main__":
    # print("\nSerial")
    # start = time.time()
    # insertor = InsertFromJsonl(app_run, tws_probs_all, mode="serial")

    # to_update = []
    # for tw_flat in tqdm.tqdm(tws_flat):
    #     tweet = insertor.check_in_db(tw_flat)
    #     if tweet is not None:
    #         to_update.append(tweet)

    # # print(to_update)
    # print(len(to_update))
    # print(f"Took {time.time() - start}s")
    # print(insertor.idx_found)

    print("Multiproc")
    start = time.time()
    insertor_multi = InsertFromJsonl(app_run, tws_probs_all, mode="multiproc")

    with multiprocessing.Pool() as pool:
        to_update = process_map(insertor_multi.check_in_db, tws_flat, chunksize=2)
    to_update = [el for el in to_update if el is not None]

    # for tw in up:
    #     print(tw)

    print(len(to_update))
    print(f"Took {time.time() - start}s")

#%%
# If no more issues, update in db.
with db:
    fields = ["tweet_id", "url", "created_at"]
    # updated = db.update_many(fields, "tweet_id", to_update)

    # Update single row at a time, easier to catch problems
    for tw in tqdm.tqdm(to_update):
        db.update(fields, "tweet_id", tw)

# %%
# WHO
insertor = InsertFromJsonl(app_run, tws_probs_all, mode="serial")
who_ha = [up[-1] for up in to_update]
df_who = df[df["handle"] == "@WHO"]
df_prob_who = df_who[~df_who["tweet_id"].isin(who_ha)]

#%%
def get_correct_infos(correct_line):
    print(Helpers.twitter_to_db_time(tws_flat[correct_line]["created_at"]))
    print(Helpers.build_tweet_url(tws_flat[correct_line]["id"], "@WHO"))


# https://www.mobilefish.com/services/unicode_escape_sequence_converter/unicode_escape_sequence_converter.php useful

# %%
# Issue
# lines 10287, 10725
# hash 7330477287
# 2 tweets with the same beginning exist
# use date to make manual decision
# -> real tweet is 1247835559881572357
# -> manually correct in db
tw_flat = tws_flat[10286]["text"]  # 1250717396073099264
tw_flat = tws_flat[10724]["text"]  # 1247835559881572357

tw_db = 'RT @WHOAFRO: Join us on Twitter tomorrow at 13:00 (CAT) for a LIVE media briefing on #COVID19 in the African Region with world-leading expe…\xa0Apr 08, 2020\xa0'

print(insertor.preprocess(tw_flat))
print(insertor.preprocess(tw_db))
print(insertor.preprocess(tw_flat) == insertor.preprocess(tw_db))

# %%
# Issue
# line 10269
# hash 3661430736 tweet_id 1250811001970319362
# -> manually correct in db

tw_flat = tws_flat[10268]["text"]  # 1250811001970319362
tw_db = 'Here are some ideas from @FIFAcom for you to #BeActive & remain healthy during #COVID19! Join our #HeathlyAtHome challenge & share with us your favorite workout moves at home!\xa0https://t.co/59qzKL0qvU\xa0Apr 16, 2020\xa0'

print(insertor.preprocess(tw_flat))
print(insertor.preprocess(tw_db))
print(insertor.preprocess(tw_flat) == insertor.preprocess(tw_db))

# %%
# Issue
# hash 9183277323 tweet_id 1250811001970319362
# lines 9044 10276
# diff is a \n, which gets stripped
# -> real is 1250784508519108608
tw_flat = tws_flat[9043]["text"]  # 1261995580060053504
tw_flat = tws_flat[10275]["text"]  # 1250784508519108608
tw_db ='Let’s grab your book 📚 & watch @HowardDonald reading “My Hero is You, How kids can fight #COVID19!” Books:\xa0https://t.co/E8EBarFxqU\xa0#HealthyAtHome #coronavirus\xa0Apr 16, 2020\xa0'

print(insertor.preprocess(tw_flat))
print(insertor.preprocess(tw_db))
print(insertor.preprocess(tw_flat) == insertor.preprocess(tw_db))

# %%
# Issue
# hash 9085616260 tweet_id 1257065499776647169
# lines  8080 9576
# same text diff dates
# -> real is 1257065499776647169
tw_flat = tws_flat[8079]["text"]  # 1269974797318062081
tw_flat = tws_flat[9575]["text"]  # 1257065499776647169
tw_db = 'Make your own avatar using @Genies & support WHO’s #StaySafe initiative with personalized stickers you can use on messaging & social media! To make your Genie, simply download the Genies app from App Store and Android #COVID19 #coronavirus\xa0https://t.co/KjinQhhjs7\xa0May 03, 2020\xa0'

print(insertor.preprocess(tw_flat))
print(insertor.preprocess(tw_db))
print(insertor.preprocess(tw_flat) == insertor.preprocess(tw_db))

# %%
# Issue
# hash 6530936744 tweet_id 1257959657479581704
# line 9447
# Twitonomy was stripping "clean/aseptic" to "cleaseptic"
# -> corrected original text in db
tw_flat = tws_flat[9446]["text"]
tw_db = 'RT @WHOWPRO: 5 moments for #HandHygiene: 1. Before touching a patient 2. Before a cleaseptic procedure 3. After body fluid exposure…\xa0May 06, 2020\xa0'

print(insertor.preprocess(tw_flat))
print(insertor.preprocess(tw_db))
print(insertor.preprocess(tw_flat) == insertor.preprocess(tw_db))

# %%
# Issue
# hash 7245775725 tweet_id
# line 8044 9396
# only diff is date
# -> real is 1270070716331950086
# -> manually corrected in db

tw_flat = tws_flat[8043]["text"]  # 1270070716331950086
tw_flat = tws_flat[9395]["text"]  # 1258475306425257984
tw_db = 'Make your own avatar using @Genies and spread WHO’s #HealthyAtHome messages with personalised stickers that you can use on messaging & social media! To make your own, simply download the Genies app from App Store or Play Store. #COVID19 #coronavirus\xa0https://t.co/DhWJaeJU7c\xa0May 07, 2020\xa0'

print(insertor.preprocess(tw_flat))
print(insertor.preprocess(tw_db))
print(insertor.preprocess(tw_flat) == insertor.preprocess(tw_db))

# %%
# Issue
# hash 9369928032 tweet_id 1260587043790471170
# line 8074 9214
# only diff is date
# -> real is 1260587043790471170
# -> manually corrected
tw_flat = tws_flat[8073]["text"]  # 1269993933951598593
tw_flat = tws_flat[9213]["text"]  # 1260587043790471170
tw_db = 'RT @WHO: Assista à conferência de imprensa da OMS para conhecer as últimas informações sobre a resposta mundial à COVID19, na sua própria l…\xa0May 13, 2020\xa0'

print(insertor.preprocess(tw_flat))
print(insertor.preprocess(tw_db))
print(insertor.preprocess(tw_flat) == insertor.preprocess(tw_db))


# %%
# Issue
# hash 1239883852
# line 8077 9217
# only diff is date
# -> real is 1260584060558204937
# -> manually corrected
tw_flat = tws_flat[8076]["text"]  #
tw_flat = tws_flat[9216]["text"]  # 1260584060558204937
tw_db = 'RT @WHO: Regardez la conférence de presse de @WHO pour obtenir les dernières informations sur la riposte mondiale à la #COVID19 dans votre…\xa0May 13, 2020\xa0'

print(insertor.preprocess(tw_flat))
print(insertor.preprocess(tw_db))
print(insertor.preprocess(tw_flat) == insertor.preprocess(tw_db))

# %%
# Issue
# hash 5348925980
# line 8076  9216
# only diff is date
# -> real is 1260584722742345728
# -> manually corrected
# tw_flat = tws_flat[8075]["text"]  #
tw_flat = tws_flat[9215]["text"]  # 1260584722742345728
tw_db = 'RT @WHO: Для получения самой актуальной информации о глобальной борьбе с #COVID19 следите за пресс-конференциями ВОЗ (@WHO) на вашем родном…\xa0May 13, 2020\xa0'

print(insertor.preprocess(tw_flat))
print(insertor.preprocess(tw_db))
print(insertor.preprocess(tw_flat) == insertor.preprocess(tw_db))

# %%
# Issue
# hash 4010458404
# line 8072 9213
# only diff is date
# -> real is 1260587319905718275
# -> manually corrected
tw_flat = tws_flat[8071]["text"]  #
tw_flat = tws_flat[9212]["text"]  # 1260587319905718275
tw_db = 'RT @WHO: विश्व स्वास्थय संगठन (WHO) के #COVID19 प्रेस कांफ्रेंस से जुड़ें हर सोमवार, बुधवार और शुक्रवार को रात 8.30 बजे (IST) क्लिक करें h…\xa0May 13, 2020\xa0'

print(insertor.preprocess(tw_flat))
print(insertor.preprocess(tw_db))
print(insertor.preprocess(tw_flat) == insertor.preprocess(tw_db))

# %%
# Issue
# hash 8502062226
# line 8039
# -> manually corrected
tw_flat = tws_flat[8038]["text"]
tw_db = 'The @Minions & Gru are supporting WHO in making sure people of all ages stay safe and healthy during this #COVID19 pandemic: 💛 physically distancing 💛 being active at home 💛 being kind to each other\xa0https://t.co/yLAKK2veHy\xa0May 27, 2020\xa0'

print(insertor.preprocess(tw_flat))
print(insertor.preprocess(tw_db))
print(insertor.preprocess(tw_flat) == insertor.preprocess(tw_db))

# %%
# Issue
# hash 4714318639
# line 8038
# ->  manually corrected
tw_flat = tws_flat[8037]["text"]
tw_db = 'Wash your hands for 40 seconds and sing along "Wash Wash Wash Your Hands" with @peppapig and and her friends, Rebecca Rabbit and Mr Badger! Washing your hands can help to stop the spread of #COVID19\xa0https://t.co/tE2kdLUiz7\xa0Jun 03, 2020\xa0'

print(insertor.preprocess(tw_flat))
print(insertor.preprocess(tw_db))
print(insertor.preprocess(tw_flat) == insertor.preprocess(tw_db))


# %%
# @UN
un_ha = [up[-1] for up in to_update]
df_un = df[df["handle"] == "@UN"]
df_prob_un = df_un[~df_un["tweet_id"].isin(un_ha)]


# %%
# @DrTedros
dr_ha = [up[-1] for up in to_update]
df_dr = df[df["handle"] == "@DrTedros"]
df_prob_dr = df_dr[~df_dr["tweet_id"].isin(dr_ha)]

# %%
# Issue
# line 6154
# hash 1219889572 tweet_id 1235276123426213889
# -> stripped [:140] now only 2 issues
# -> above solution not working for other tweets.
# the difference between the 2 is the URLs at the end.
# -> link manually corrected
tw_flat = "Hey @TheEllenShow, thank you so much for the hand washing lesson &amp; using @WHO guidance... but you missed a few spots.🙂 I will send you a video on how public health geeks like me do it 🤓. We need your continued help in communicating #coronavirus tips. Let's stay in touch! https://t.co/xhkhSHP8vQ https://t.co/hW89aqFeec"
tw_db = "Hey @TheEllenShow, thank you so much for the hand washing lesson & using @WHO guidance... but you missed a few spots.🙂 I will send you a video on how public health geeks like me do it 🤓. We need your continued help in communicating #coronavirus tips. Let's stay in touch!\xa0https://t.co/hW89aqFeec\xa0https://t.co/xhkhSHP8vQ\xa0Mar 04, 2020\xa0"
print(insertor_multi.preprocess(tw_flat))
print(insertor_multi.preprocess(tw_db))
print(insertor_multi.preprocess(tw_flat) == insertor_multi.preprocess(tw_db))

# %%
# Issue
# lines 5947 and 5949
# hash 1402152831 tweet_id
# Actually 2 different tweets, problem because we compare
# stripped tweets. -> remove stripping
tw_flat = 'Great to have your strong voice, @Alissonbecker, joining @WHO in such a critical moment to keep the world safe from #COVID19. Thank you for following our advice and sharing it further. \n\nhttps://t.co/aSj9AY3FHE'
tw_db = 'Great to have your strong voice, @Alissonbecker, joining @WHO in such a critical moment to keep the world safe from #COVID19. Thank you for following our advice and sharing it further.\xa0https://t.co/8lxAr0lDCi\xa0Mar 12, 2020\xa0'

print(insertor_multi.preprocess(tw_flat))
print(insertor_multi.preprocess(tw_db))
print(insertor_multi.preprocess(tw_flat) == insertor_multi.preprocess(tw_db))

# %%
# Issue
# line 6134
# hash 4298891474 tweet_id 1235652211889209344
# link manually corrected in db

tw_flat = 'What a cute way to express love during #COVID19 (hoping that soap was used 🙂)!\n\nTo be safe from #coronavirus and protect your loved ones, wash your hands regularly with soap and water, or rub them with alcohol-based solutions. https://t.co/UZOVakAK5C https://t.co/kZLPimKRMh'
tw_db = 'What a cute way to express love during #COVID19 (hoping that soap was used 🙂)! To be safe from #coronavirus and protect your loved ones, wash your hands regularly with soap and water, or rub them with alcohol-based solutions.\xa0https://t.co/kZLPimKRMh\xa0https://t.co/UZOVakAK5C\xa0Mar 05, 2020\xa0'

print(insertor_multi.preprocess(tw_flat))
print(insertor_multi.preprocess(tw_db))
print(insertor_multi.preprocess(tw_flat) == insertor_multi.preprocess(tw_db))

# %%
# Issue
# lines 5973
# hash 1356039882 tweet_id 1237800182235922434
# link manually corrected in db

tw_flat = tws_flat[5972]["text"]
tw_db = '.@WHO is deeply concerned by the alarming levels of the #coronavirus spread, severity & inaction, & expects to see the number of cases, deaths & affected countries climb even higher. Therefore, we made the assessment that #COVID19 can be characterized as a pandemic.\xa0https://t.co/97XSmyigME\xa0https://t.co/gSqFm947D8\xa0Mar 11, 2020\xa0'

print(insertor_multi.preprocess(tw_flat))
print(insertor_multi.preprocess(tw_db))
print(insertor_multi.preprocess(tw_flat) == insertor_multi.preprocess(tw_db))

# %%
# Issue
# lines 5877 5878 5879
# hash 1063965577 tweet_id 1237800182235922434
# & not is stripped to &not which is the HTML entity ¬
# -> html.unescape before stripping spaces

tw_flat = tws_flat[5878]["text"]
tw_db = 'Being Ready for #COVID19 means acting as part of a local & global community. This means not hoarding vital items, like gloves & face masks. It means sharing & not stockpiling. It means helping others & not shunning. Working together is key to fighting #coronavirus.\xa0Mar 15, 2020\xa0'

print(insertor_multi.preprocess(tw_flat))
print(insertor_multi.preprocess(tw_db))
print(insertor_multi.preprocess(tw_flat) == insertor_multi.preprocess(tw_db))

#%%
# @enmarchefr
en_ha = [up[-1] for up in to_update]
df_en = df[df["handle"] == "@enmarchefr"]
df_prob_en = df_en[~df_en["tweet_id"].isin(en_ha)]
# No issue

# %%
# Issue
# check @RY ...:


# %%
# Fix duplicates (remove if theme_hardcoded != "0")
idx_search = []
count = 0
with db:
    for tw_id in tqdm.tqdm(idx_search):
        tw = db.get_tweet_by_id(tw_id)
        try:
            fields = tw[-5:]  # Also check that no coded tweets are deleted
        except Exception as e:  # If tweet_id not in db
            print(e)
        if not any(fields):
            count += 1
            # print("delete", fields)
            db.delete_by_id(tw_id)
print(count)
