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

        with open('../resources/actors_url.txt', 'r') as f:
            urls = f.readlines()
            urls = [url.strip('\n') for url in urls]
        return urls

    @staticmethod
    def dynamic_text(text: str):
        """
        Dynamically write text to the terminal
        """

        print(text, end="\r")