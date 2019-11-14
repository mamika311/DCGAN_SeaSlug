# pip install icrawler

from icrawler.builtin import GoogleImageCrawler

crawler = GoogleImageCrawler(storage={"root_dir": "image"})
crawler.crawl(keyword="ウミウシ", max_num=100)
