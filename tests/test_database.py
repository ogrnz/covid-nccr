# pylint: skip-file

import sys, os

sys.path.append(os.path.abspath(os.path.join("..", "src")))

import unittest

unittest.TestLoader.sortTestMethodsUsing = None

from common.app import App
from common.database import Database
from common.helpers import Helpers


class TestDatabase(unittest.TestCase):

    tweets = [
        (
            "1211853090552332289",
            0,
            "31/12/2019 03:34:26",
            "@WHOSEARO",
            "WHO South-East Asia",
            "RT @WHOSEARO: ALWAYS seek the advice of a qualified healthcare professional befortaking #antibiotics 💊\n\nAntibiotics can be used as part o…",
            "ALWAYS seek the advice of a qualified healthcare professional before takin#antibiotics 💊\n\nAntibiotics can be used as part of treatment for many infectionsincl.:\n-Pneumonia\n-Syphilis\n-TB\n\nBut remember: They don't treat viral infectionslike colds &amp; flu\n\n#TuesdayThoughts https://t.co/PsYZIbcc6l",
            "https://twitter.com/WHOSEARO/status/1211853090552332289",
            "Retweet",
            "56",
            "0",
            None,
            None,
            None,
            None,
            None,
        ),
        (
            "8826085549",
            1,
            "22/05/2020",
            "@WHO",
            "World Health Organization (WHO)",
            None,
            "Live from #EB147. #COVID19\xa0https://t.co/RuQaUprrWR\xa0May 22, 2020\xa0",
            None,
            "Reply",
            None,
            None,
            "604",
            None,
            None,
            None,
            None,
        ),
        (
            "1427204514906529792",
            0,
            "16/08/2021 09:44:07",
            "@NHSuk",
            "NHS",
            None,
            "Discomfort or diarrhoea for three weeks or more, or blood in your pee – even jusonce, could be a sign of cancer. It’s probably nothing serious, but finding canceearly makes it more treatable.\n\nYour NHS wants to see you. Just contact your Gpractice. \n\nhttps://t.co/hivbfrsuIl https://t.co/EjCeR0uoj5",
            "https://twitter.com/NHSuk/status/1427204514906529792",
            "New",
            "6",
            "8",
            None,
            None,
            None,
            None,
            None,
        ),
    ]

    tweet_newer = (
        "1",
        0,
        "16/08/2021 09:45:07",
        "@NHSuk",
        "NHS",
        None,
        "Newer (fake) tweet from @NHSuk.",
        "https://twitter.com/NHSuk/status/1427204514906529792",
        "Reply",
        "6",
        "8",
        None,
        None,
        None,
        None,
        None,
    )

    tweets_duplicate = [
        (
            "1211853090552332289",
            0,
            "31/12/2019 03:34:26",
            "@WHOSEARO",
            "WHO South-East Asia",
            "UPDATED",
            "ALWAYS seek the advice of a qualified healthcare professional before takin#antibiotics 💊\n\nAntibiotics can be used as part of treatment for many infectionsincl.:\n-Pneumonia\n-Syphilis\n-TB\n\nBut remember: They don't treat viral infectionslike colds &amp; flu\n\n#TuesdayThoughts https://t.co/PsYZIbcc6l",
            "https://twitter.com/WHOSEARO/status/1211853090552332289",
            "Retweet",
            "56",
            "0",
            None,
            None,
            None,
            None,
            None,
        ),
        (
            "8826085549",
            1,
            "22/05/2020",
            "@WHO",
            "World Health Organization (WHO)",
            "UPDATED",
            "Live from #EB147. #COVID19\xa0https://t.co/RuQaUprrWR\xa0May 22, 2020\xa0",
            None,
            "New",
            None,
            None,
            "604",
            None,
            None,
            None,
            None,
        ),
    ]

    db_name = "test/test_database.db"

    @classmethod
    def setUpClass(cls):
        cls.app = App(debug=False)
        cls.db_path = f"{cls.app.root_dir}/database/{cls.db_name}"
        cls.del_db(cls)
        cls.db = Database(db_name=cls.db_name, app=cls.app)

    def del_db(self):
        """
        Delete test database if it exists to avoid failed tests.
        """

        if os.path.isfile(self.db_path):
            os.remove(self.db_path)

    def test_a_insert_many(self):
        with self.db:
            self.db.insert_many(self.tweets)
            tweets = self.db.get_all_tweets()

        self.assertEqual(len(tweets), len(self.tweets))

    def test_b_get_all_tweets(self):
        with self.db:
            tweets = self.db.get_all_tweets()

        self.assertEqual(len(tweets), len(self.tweets))

    def test_c_insert_or_replace_many(self):
        with self.db:
            self.db.insert_or_replace_many(self.tweets_duplicate)
            tweets = [
                self.db.get_tweet_by_id(idx)
                for idx in ["8826085549", "1211853090552332289"]
            ]

        self.assertEqual(len(tweets), len(self.tweets_duplicate))
        self.assertEqual(tweets[0][0][5], "UPDATED")
        self.assertEqual(tweets[1][0][5], "UPDATED")

    def test_d_insert_tweet(self):
        with self.db:
            self.db.insert_tweet(self.tweet_newer)
            inserted_tweet = self.db.get_tweet_by_id("1")

        self.assertEqual(inserted_tweet[0][0], "1")
        # self.assertEqual(inserted_tweet[0][0], 1)

    def test_e_get_fields(self):
        fields = ["tweet_id"]
        with self.db:
            tweets = self.db.get_fields(fields=fields)
            tweets_covid = self.db.get_fields(fields=fields, only_covid=True)
            tweets_limit = self.db.get_fields(fields=fields, limit=2)

        [self.assertEqual(len(tweet), len(fields)) for tweet in tweets]
        self.assertEqual(len(tweets_covid), 1)
        self.assertEqual(tweets_covid[0][0], "8826085549")
        # self.assertEqual(tweets_covid[0][0], 8826085549)
        self.assertEqual(len(tweets_limit), 2)

    def test_f_get_by_type(self):
        types = ["Retweet", "New", "Reply"]
        with self.db:
            tweets = [self.db.get_by_type(t) for t in types]

        self.assertEqual(len(tweets[0]), 1)
        self.assertEqual(len(tweets[1]), 2)
        self.assertEqual(len(tweets[2]), 1)

    def test_g_update_tweet_by_id(self):
        with self.db:
            self.db.update_tweet_by_id(1, "text", "UPDATED")
            updated = self.db.get_tweet_by_id("1")

        self.assertEqual(updated[0][6], "UPDATED")

    def test_h_update_theme_many(self):
        to_update = [(1, "1427204514906529792"), (1, "1211853090552332289")]

        with self.db:
            self.db.update_theme_many(to_update)
            updated = [
                self.db.get_tweet_by_id(idx)
                for idx in ["1427204514906529792", "1211853090552332289"]
            ]

        self.assertEqual(updated[0][0][1], 1)
        self.assertEqual(updated[1][0][1], 1)

    def test_i_update_many(self):
        to_update = [(111, "1427204514906529792"), (222, "1211853090552332289")]

        with self.db:
            self.db.update_many("frame", to_update)
            updated = [
                self.db.get_tweet_by_id(idx)
                for idx in ["1427204514906529792", "1211853090552332289"]
            ]
        self.assertEqual(updated[0][0][-2], "111")
        self.assertEqual(updated[1][0][-2], "222")

    def test_j_get_last_id_by_handle(self):
        with self.db:
            last_id = self.db.get_last_id_by_handle("NHSuk")

        self.assertEqual(last_id[0], "1427204514906529792")
        # self.assertEqual(last_id[0], 1427204514906529792)
