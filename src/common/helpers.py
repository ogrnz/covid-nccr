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
    def print_timer(elapsed):
        """
        Print elapsed time in terminal
        """
        print(f"\nDone in {round(elapsed, 1)} s")


if __name__ == "__main__":
    pass
