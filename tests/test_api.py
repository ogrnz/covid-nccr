# pylint: skip-file

import sys, os

sys.path.append(os.path.abspath(os.path.join("..", "src")))

import unittest

from common.app import App
from common.api import Api


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = App(debug=False)
        cls.api = Api(self.app)

    def test_api_connection(self):
        self.api = Api(self.app)
        self.assertTrue(self.api.connected)

    def test_get_tweets(self):
        """
        This account was selected because there is only 1 posted tweet.

        !! Won't work if the account is deleted. !!
        """

        tweets = self.api.get_tweets("brexitparty_uk")
        self.assertGreater(len(tweets), 0)

    def test_get_tweets_before_oldest(self):
        """
        Since we do not care about tweets older than 31/12/2019 (tweet_id ==
        1211913001147740161), we should not get back any tweets.

        !! This test won't work if I post a new tweet. !!
        """

        tweets = self.api.get_tweets("opotrac", last_id=1211913001147740161)
        self.assertEqual(len(tweets), 0)

    def test_get_tweets_by_ids(self):
        """
        This test also checks if our academic access is still valid by
        retrieving an old tweet.
        """

        tws_list = [20, -10, 1211942570244231169]
        tweets = self.api.get_tweets_by_ids(tws_list)

        self.assertEqual(len(tws_list), len(tweets))
        self.assertIsNotNone(tweets.iloc[0, :][2])
        self.assertIsNone(tweets.iloc[1, :][2])

    def test_get_complete_tweets_by_ids(self):
        tws_list = [20, -10, 1425382732230807561]
        tweets = self.api.get_complete_tweets_by_ids(tws_list)

        self.assertGreater(len(tweets.loc[3, "fulltext"]), 140)
