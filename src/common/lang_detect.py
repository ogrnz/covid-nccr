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
        return lang.replace("__label__", "")

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
