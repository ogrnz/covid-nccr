import os
import re

with open('actors_url.txt', 'r') as f:
    urls = f.readlines()

urls = [url.strip('\n') for url in urls]

path = os.path.join(os.getcwd(), 'tweets-data')

for url in urls:
    # strip urls 
    screen_name = url[19:len(url)]
    print(url[19:len(url)])

    try:
        dir_path = os.path.join(path, screen_name) 
        os.mkdir(dir_path)
    except OSError:
        print (f"Creation of the directory {path} failed")
