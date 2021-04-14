"""
Export xlsx file to server via webdav
"""

from webdav3.client import Client
from webdav3.exceptions import WebDavException

from common.app import App


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

    print(f"Uploading {filename} to remote server")
    try:
        client.upload_sync(remote_path=dest + filename, local_path=orig + filename)
        print(f"Done uploading in {dest}")
    except WebDavException as error:
        print("Error", error)


if __name__ == "__main__":
    app_run = App(debug=False)
    FILES = ["Covid-tweets-2021-04-14.xlsx", "Covid-tweets-2021-04-14.xlsx"]

    for FILE in FILES:
        main(FILE, app=app_run)
