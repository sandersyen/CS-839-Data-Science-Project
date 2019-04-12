import os

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests import ConnectionError, RequestException

headers = ['Name', 'Author', 'Format', 'Page', 'Language', 'ISBN_10', 'ISBN_13']

tag = {'header': '<div class="item-info">',
       'Name': ('<h1 itemprop="name">', '</h1>'),
       'Author': ('<span itemprop="name">', '</span>'),
       'Format': ('<label>Format</label>\n<span>', '|'),
       'Page': ('<span itemprop="numberOfPages">', 'pages'),
       'Language': ('<label>Language</label>\n<span>', '</span>'),
       'ISBN_10': ('<label>ISBN10</label>\n<span>', '</span>'),
       'ISBN_13': ('<span itemprop="isbn">', '</span>')}


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
    data = dict()                           # data frame to store desired info

    # information extraction
    for header in headers:
        index = extract_specific_feature(feature=header, html=html, index=index,
                                         data=data, i1=tag[header][0], i2=tag[header][1])
    return df.append(data, ignore_index=True)


def main():
    urls = pd.read_csv('data/urls_source1.csv')                 # load the urls got

    # check if the file exists. create file if not exists
    if not os.path.isfile('data/data_source1.csv'):
        with open('data/data_source1.csv', 'a') as f:
            pd.DataFrame(columns=headers).to_csv(f, sep=',', index=False, header=True)

    for i in range(len(urls)):
        df = pd.DataFrame(columns=headers)              # data frame for storing info for the current url
        print(i, urls['url'][i])
        # use try except to prevent from some specific links whose info cannot be extracted
        try:
            df = info_extraction(urls['url'][i], df)
        except ValueError:
            print(str(i) + ' has value error')
            continue

        # append result to the output file
        with open('data/data_source1.csv', 'a') as f:
            df.to_csv(f, sep=',', index=False, header=False)

    # remove duplication
    df = pd.read_csv('data/data_source1.csv')
    df.drop_duplicates(subset=['Name', 'Author'], keep="first", inplace=True)
    df.to_csv('data/data_source1.csv', index=True)


if __name__ == "__main__":
    main()
