# pylint: skip-file

import requests
import time

with open("actors_url.txt", "r") as f:
    urls = f.readlines()

urls = [url.strip("\n") for url in urls]
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:85.0) Gecko/20100101 Firefox/85.0"
}

for url in urls:
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        print(r, r.status_code, r.text)
    time.sleep(0.5)

# All actors' URL are valid
