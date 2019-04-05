import os
from unidecode import unidecode
import re

def read_data(articles):
    """
    read the file and append it to the articles
    :param articles: a list for all articles
    :return: None
    """

    def files(path):
        """
        find all file in a given path and yield the files' paths
        :param path: the path to the location of files
        :return: files' paths
        """
        for f in os.listdir(path):
            if len(f.split('.')[0]) == 3 and f.split('.')[1] == "txt" and os.path.isfile(os.path.join(path, f)):
                yield path + "/" + f, f.split('.')[0]

    for file_path, filename in files("data"):
        articles.append((filename, unidecode(file(file_path, 'r').read().decode("UTF-8"))))


def data_split(articles):
    """
    split data into two data-sets (training and testing)
    :param articles: a list of articles
    :return: two lists for two data-sets (training and testing)
    """
    train_set, test_set = [], []

    for i in range(0, len(articles), 3):
        train_set.append(articles[i])
        train_set.append(articles[i+1])
        test_set.append(articles[i+2])

    return train_set, test_set


def label_extraction_takeoff(paragraphs, count, labels=None):
    """
    Take off the label <person> and </person> and return the paragraph without labels
    :param paragraphs: string input data with <person></person> labels
    :param count: number of labels in articles
    :param labels: a set which contains all label among all input data
    :return: new paragraohs without labels, number of labels in articles
    """
    LABEL, LABEL_END = "<person>", "</person>"
    index, new_paragraph = 0, ""
    filename = paragraphs[0]
    paragraphs = paragraphs[1]

    while index < len(paragraphs):
        # find the index of the closest LABEL
        found = paragraphs.find(LABEL, index)

        # if the label is found
        if found != -1:
            # find the index (location) of the end of label
            found_end = paragraphs.find(LABEL_END, found)
            # append label to the return variable new_paragraph
            new_paragraph += paragraphs[index:found] + paragraphs[found+len(LABEL):found_end]

            # if labels is not None, add the label into it
            if labels is not None:
                labels.add(re.sub('[?;!@#$(){}\\,\\."]', '', paragraphs[found+len(LABEL):found_end]))

            # update the current index
            index = found_end + len(LABEL_END)
            count += 1

        else:
            new_paragraph += paragraphs[index:]
            break

    return (filename, new_paragraph), count, labels
