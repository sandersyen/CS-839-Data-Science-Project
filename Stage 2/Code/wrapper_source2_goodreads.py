import pandas as pd
import requests
import os
from bs4 import BeautifulSoup
from requests import ConnectionError, RequestException

headers = ['Name', 'Author', 'Format', 'Pages', 'ISBN_10', 'ISBN_13', 'Language']

tag = {'header': '<div class="last col" id="metacol">',
       'Name': ('<h1 class="gr-h1 gr-h1--serif" id="bookTitle" itemprop="name">', '</h1>'),
       'Author': ('<span itemprop="name">', '</span>'),
       'Format': ('<span itemprop="bookFormat">', '</span>'),
       'Pages': ('<span itemprop="numberOfPages">', 'pages'),
       'Published Date': ('Published', 'by'),
       'ISBN_10': ('ISBN</div>\n<div class="infoBoxRowItem">', '<span class="greyText">'),
       'ISBN_13': ('ISBN13: <span itemprop="isbn">', '</span>'),
       'Language': ('<div class="infoBoxRowItem" itemprop="inLanguage">', '</div>')}

def extract_specific_feature(feature, html, index, data, i1, i2):
    """
    Extract data of given feature and store in "data"
    :param feature: the feature of desired data
    :param html: the entire web-page text
    :param index: the current start index (point)
    :param data: data frame to store info
    :param i1: information tag 1 (head)
    :param i2: information tag 2 (tail)
    :return: updated index (after i2)
    """
    start_index = html.index(i1, index) + len(i1)
    end_index = html.index(i2, start_index)
    data[feature] = html[start_index:end_index].strip()
    return end_index + len(i2)

def info_extraction(url, df):
    """
    Extract information from a url
    :param url: the web page to extract info from
    :param df: data frame to store all tuples
    :return: updated df
    """
    # request html text from the url
    try:
        html = requests.get(url).text
        html = str(BeautifulSoup(html, 'html.parser'))
    except(IOError, ValueError, TypeError, ConnectionError, RequestException):
        print("Web-page " + url + " error")
        return None

    # initialise
    index = html.index(tag['header'])       # index of current position in html
    # information extraction
    data = dict()                           # data frame to store desired info
    for header in headers:
        try:
            index = extract_specific_feature(feature=header, html=html, index=index,
                                             data=data, i1=tag[header][0], i2=tag[header][1])
        except(IOError, ValueError, TypeError, ConnectionError, RequestException):
            data[header] = "Unknown"
    return df.append(data, ignore_index=True)

def main():
    urls = pd.read_csv('data/urls_source2_goodreads.csv')                 # load the urls got
    if not os.path.isfile('data/data_source2.csv'): 
        temp = pd.DataFrame(columns=headers)              # data frame for storing info for the current url   
        with open('data/data_source2.csv', 'a') as f:
            temp.to_csv(f, sep=',', index=False, header=True)
    else:
        df = pd.read_csv('data/data_source2.csv', index_col=0)
        df.to_csv('data/data_source2.csv', index=False, header=True)

    for i in range(len(urls['url'])):
        df = pd.DataFrame(columns=headers)              # data frame for storing info for the current url    
        print(i, urls['url'][i])
        # use try except to prevent from some specific links whose info cannot be extracted
        try:
            df = info_extraction(urls['url'][i], df)
        except ValueError:
            print(str(i) + ' has value error')
            continue

        # append result to the output file
        with open('data/data_source2.csv', 'a') as f:
            df.to_csv(f, sep=',', index=False, header=False)

    df = pd.read_csv('data/data_source2.csv')
    # maybe there is other way to do this.
    df.drop_duplicates(subset=['Name', 'Author'], keep="first", inplace=True)
    df.index.name = "ID"
    df.to_csv('data/data_source2.csv')

if __name__ == "__main__":
    main()
