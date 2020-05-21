from urllib.request import urlopen
from bs4 import BeautifulSoup
import re

open_set = set()  # meet
close_set = set()  # visted
pattern = re.compile("latimes.com")

def get_links(pageUrl):
    close_set.add(pageUrl)
    html = urlopen(pageUrl)
    bsObj = BeautifulSoup(html, "html5lib")
    for link in bsObj.find_all('a', href=pattern):
        nextUrl = link.attrs['href']
        if nextUrl not in open_set and nextUrl not in close_set:
            print(nextUrl)
            open_set.add(nextUrl)

get_links("http://www.latimes.com")
while len(open_set) != 0:
    get_links(open_set.pop())

