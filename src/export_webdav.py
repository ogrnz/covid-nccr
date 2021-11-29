"""
Export xlsx file to server via webdav
"""

import os
import logging

from webdav3.client import Client
from webdav3.exceptions import WebDavException

from common.app import App

log = logging.getLogger(os.path.basename(__file__))


def main(filename, app: App):
    """
    Main function
    """

    options = {
        "webdav_hostname": app.webdav_server,
        "webdav_login": app.webdav_login,
        "webdav_password": app.webdav_pwd,
    }

    dest = "Sent files/covid/"
    orig = f"{app.root_dir}/database/xlsx/"

    client = Client(options)

    log.info(f"Uploading {filename} to remote server")
    try:
        client.upload_sync(remote_path=dest + filename, local_path=orig + filename)
        log.info(f"Done uploading in {dest}")
    except WebDavException as error:
        log.warning(f"Error {error}")


if __name__ == "__main__":
    app_run = App(debug=False)
    FILES = ["Covid-tweets-2021-04-14.xlsx", "Covid-tweets-2021-04-14.xlsx"]

    for FILE in FILES:
        main(FILE, app=app_run)
