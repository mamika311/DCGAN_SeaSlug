from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import os, time, sys

# APキーIの情報
key = "0b83ac06083441343d8110d607bf4c80"
secret = "2daf4beaf84e6acf"
wait_time = 1

# 保存フォルダの指定
savedir = "./image"

flickr = FlickrAPI(key, secret, format="parsed-json")
result = flickr.photos.search(
        per_page = 400,
        tags = "seaslug",
        media = "photos",
        sort = "relevance",
        safe_search = 1,
        extras = "url_q, licence"
)

photos = result["photos"]

# ループ処理でphotoに情報を格納する
for i, photo in enumerate(photos['photo']):
    url_q = photo["url_q"]
    filepath = savedir + "/" + photo["id"] + ".jpg"
    if os.path.exists(filepath): continue
    urlretrieve(url_q, filepath)
    time.sleep(wait_time)
