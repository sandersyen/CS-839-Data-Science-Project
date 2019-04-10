from bs4 import BeautifulSoup
import requests
import pandas as pd
from requests import ConnectionError, RequestException


url = 'https://www.bookdepository.com/bestof2018'   # website to get urls
prefix = 'https://www.bookdepository.com'           # the website prefix

df = pd.read_csv('data/urls_source1.csv')                   # load the urls got

# check if the url is available
try:
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html.parser')
except(IOError, ValueError, TypeError, ConnectionError, RequestException):
    print("Web-page " + url + " error")
    exit()

# extract all books' urls
for each in soup.select('.title a'):
    df = df.append({'url': prefix+str(each['href'])}, ignore_index=True)

df.drop_duplicates(subset="url", keep="first", inplace=True)    # remove duplication
df.to_csv(path_or_buf='data/urls_source1.csv', index=False)             # save urls
