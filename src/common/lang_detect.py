"""
Detect language (en, fr, other) of a tweet.
"""

import os
import logging

from googletrans import Translator
import fasttext

log = logging.getLogger(os.path.basename(__file__))

translator = Translator()
fasttext_bin = os.path.join("..", "..", "database", "lid.176.bin")
fmodel = fasttext.load_model(fasttext_bin)

def lang_detect(txt, threshold=0.85):
    """
    Detect tweet language
    returns None if confidence lvl < threshold
    """

    if txt is None:
        return None

    txt = txt.replace("\n", " ")
    detection = translator.detect(txt)
    if isinstance(detection.lang, list) or detection.confidence < threshold:
        return None
    else:
        return detection.lang

def en_detect(txt, threshold=0.85):
    """
    Detect if tweet language is en with confidence <= threshold
    """

    if txt is None:
        return False

    txt = txt.replace("\n", " ")
    detection = translator.detect(txt)
    if isinstance(detection.lang, list):
        return False
    elif detection.confidence < threshold or detection.lang != "en":
        return False
    else:
        return True

def fasttext_detect(txt):
    """
    Fasttext raw implementation
    """

    if txt is None:
        return None

    txt = txt.replace("\n", " ")
    return fmodel.predict(txt)

def fasttext_lang_detect(txt, threshold=0.5):
    """
    Detect lang of tweet with pretrained fasttext
    """

    if txt is None:
        return None

    txt = txt.replace("\n", " ")
    detection = fmodel.predict(txt)
    lang = detection[0][0]
    confidence = detection[1][0]

    if confidence < threshold:
        return None
    else:
        return lang.strip("__label__")

def fasttext_en_detect(txt, threshold=0.5):
    """
    Detect if tweet language is en with confidence <= threshold
    """

    if txt is None:
        return False

    txt = txt.replace("\n", " ")

    detection = fmodel.predict(txt)
    lang = detection[0][0]
    confidence = detection[1][0]

    return confidence >= threshold and lang == "__label__en"



#%%

if False:
    app_run = App(debug=False)
    db = Database("tweets.db", app=app_run)
    api = Api(main_app=app_run)
    data_path = os.path.join(app_run.root_dir, "interactive", "data")


    with db:
        tweets = db.get_all_tweets()
    len(tweets)

    df = pd.DataFrame(
        tweets,
        columns=Helpers.schema_cols,
    )
    df["url"].isna().sum()
    # 267 NA's -> tweets with no URL, only '@MinSoliSante' tweets (account deleted)


    identifier.classify(df["text"][0])
    # works!


    def lang_detect(txt, threshold=0.9):
        """
        Detect tweet language
        returns None if confidence lvl < threshold
        """

        if txt is None:
            return None

        txt = txt.replace("\n", " ")
        lang = identifier.classify(txt)
        if lang[1] < threshold:
            return None
        else:
            return lang[0]


    # If `text` is empty, use `old_text`
    subs = df[df["theme_hardcoded"] == "0"].head(100)
    subs["lang"] = subs.apply(
        lambda row: lang_detect(row["old_text"])
        if row["text"] is None
        else lang_detect(row["text"]),
        axis=1,
    )

    subs["lang"].unique()
    # "en", "fr", "de", "it", nan, "am"

    # For whole dataset
    tqdm.pandas()
    df["lang"] = df.progress_apply(
        lambda row: lang_detect(row["old_text"])
        if row["text"] is None
        else lang_detect(row["text"]),
        axis=1,
    )
    df["lang"].unique()


    df.to_pickle(os.path.join(data_path, "db_all_lang.pkl"))

    # If lang is not en or fr, set it to `other`
    lang_lst = ["en", "fr"]
    df["lang"] = df["lang"].progress_apply(lambda x: "other" if x not in lang_lst else x)
    df["lang"].unique()

    """
    en - 138958
    fr - 82196
    other - 19221
    """

    # df.to_pickle(os.path.join(data_path, "db_sub_lang.pkl"))

