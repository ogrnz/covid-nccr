"""
Export xlsx file to server via webdav
"""

from webdav3.client import Client
from webdav3.exceptions import WebDavException

from common.app import App


def main(app_run: App, filename):
    """
    Main function
    """

    options = {
        "webdav_hostname": app_run.webdav_server,
        "webdav_login": app_run.webdav_login,
        "webdav_password": app_run.webdav_pwd,
    }

    dest = "Sent files/covid/"
    orig = "database/xlsx/"

    client = Client(options)

    print(f"Uploading {filename} to remote server")
    try:
        client.upload_sync(remote_path=dest + filename, local_path=orig + filename)
    except WebDavException as error:
        print("Error", error)


if __name__ == "__main__":
    app = App(debug=True)
    FILENAME = "tweets-2021-02-27.xlsx"

    main(app, FILENAME)
