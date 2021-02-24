"""
Helpers module
"""


class Helpers:
    """
    Class that contains helper functions
    """

    def __init__(self):
        pass

    @staticmethod
    def get_actors_url(filename: str) -> list:
        """
        Retrieve all accounts URLs located in file
        """

        with open(f"src/resources/{filename}", "r") as file:
            urls = file.readlines()
            urls = [url.strip("\n") for url in urls]
        return urls

    @staticmethod
    def dynamic_text(text: str):
        """
        Dynamically write text to the terminal
        """

        print(text, end="\r")

    @staticmethod
    def get_screen_names(urls: list) -> list:
        """
        Returns screen_names list stripped from urls
        """

        return [url[19 : len(url)] for url in urls]

    @staticmethod
    def classify(text) -> bool:
        """
        Classify a string as being about covid or not
        """
        with open("src/resources/covid_keywords.txt", "r", encoding="utf-8") as file:
            words = [line.strip("\n") for line in file.readlines()]

        text_split = text.split()

        for txt_ele in text_split:
            return bool(txt_ele in words)


if __name__ == "__main__":
    text1 = "Hello is this tweet about covid?"
    text2 = "je ne pense pas que ce tweet concerne ce que je pense"
    text3 = "covid"

    print(Helpers.classify(text1))
    print(Helpers.classify(text2))
    print(Helpers.classify(text3))
