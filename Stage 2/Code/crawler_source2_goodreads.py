from bs4 import BeautifulSoup
import requests
import pandas as pd
from requests import ConnectionError, RequestException


url = 'https://www.goodreads.com/list/show/119296.What_We_ve_Read_So_Far_in_2018?page='   # website to get urls
prefix = 'https://www.goodreads.com'           # the website prefix

# df = pd.read_csv('data/urls_goodreads.csv')                   # load the urls got
df = pd.DataFrame(columns=["url"])              # data frame for storing info for the current url

for i in range(1, 66):
    # append result to the output file
    with open('data/urls_goodreads.csv', 'a') as f:
        df.to_csv(f, sep=',', index=False, header=False)

    # check if the url is available
    try:
        print (url + str(i))
        html = requests.get(url + str(i)).text
        soup = BeautifulSoup(html, 'html.parser')
    except(IOError, ValueError, TypeError, ConnectionError, RequestException):
        print("Web-page " + url + " error")
        exit()

    # extract all books' urls
    data = soup.findAll('a',attrs={'class':'bookTitle'})
    for a in data:
        print (a['href'])
        df = df.append({'url': prefix+str(a['href'])}, ignore_index=True)

print (df)
df.drop_duplicates(subset="url", keep="first", inplace=True)    # remove duplication
df.to_csv(path_or_buf='data/urls_goodreads.csv', index=False)             # save urls
