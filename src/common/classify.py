"""
Classify module
"""

import re


class Classifier:
    """
    Classify a text as being about covid or not
    """

    def __init__(self, keywords_file="covid_final.txt"):
        self.keywords = self.load_keywords(keywords_file)
        self.reg = self.create_reg()

    def create_reg(self):
        """
        Create regex pattern based on keyword list
        """

        return r"|".join(self.keywords)

    @staticmethod
    def load_keywords(filename: str) -> list:
        """
        Load .txt file that contains keyword list
        """
        with open(f"src/resources/{filename}", "r", encoding="utf-8") as file:
            words = [line.strip("\n") for line in file.readlines()]

        return words

    def classify(self, txt: str) -> bool:
        """
        Classify a string as being about covid or not
        """
        if re.search(self.reg, str(txt), re.I):
            return True
        else:
            return False


if __name__ == "__main__":
    tweet1 = (
        1364514826270179332,
        None,
        "Retweet",
        "24/02/2021 09:57:40",
        "@ECDC_EU",
        "ECDC",
        "RT @EU_opendata: �📣 A new #Dataset is out! \nLooking for information about #COVID19 testing rate and test positivity? �🏥\n\nCheck out the data p…",
        "📣📣 A new #Dataset out!\n\nLooking for information about #COVID19 testing rate and test positivity? �🏥\n\nCheck out the data provided by @ECDC_E �👉 https://t.co/TORp8qmhbL\n\n#EUOpenData #CovidTesting #CovidData @ECDC_Outbreaks https://t.co/gwpnzcWJ9f",
        "https://twitterom/ECDC_EU/status/1364514826270179332",
        3,
        0,
    )

    classifier = Classifier()
    # print(classifier.reg)
    # print(classifier.classify(tweet1[7]))
