import re


def generate_ngrams(filename, content, n):
    """
    Generate n-grams (with a feature whether it contains "'s") from the content
    :param filename: filename
    :param content: the whole article
    :param n: the size of n gram
    :return: generated list of n-grams, single grams
    """
    sentences = content.split(".")
    index, index2 = 0, 0
    n_grams, single_grams, single_grams2 = [], [], [] 
    for sentence in sentences:
        sections = sentence.split(",")
        for section in sections:
            parts = section.split(";")
            for part in parts:
                words = part.split()
                single_grams_temp, feature_single_quote_temp = [], []
                for i in range(len(words)):
                    words2 = words[:]
                    words2[i] = re.sub('[;@#$()\{\}:"]', '', words2[i])
                    single_grams2.append((words2[i], filename, index2, index2))
                    index2 += 1

                # first clean the data
                for i in range(len(words)):
                    # clean data by removing special characters
                    words[i] = re.sub('[?;!@#$()\{\}:\,\."]', '', words[i])

                    # for cases 's, take off 's
                    if (len(words[i]) >= 2 and words[i][-2] == "'"):
                        words[i] = words[i][:-2]
                        feature_single_quote_temp.append(1)
                    elif (len(words[i]) >= 2 and words[i][-2] == "s" and words[i][-1] == "'"):
                        words[i] = words[i][:-1]
                        feature_single_quote_temp.append(1)
                    else:
                        feature_single_quote_temp.append(0)

                    single_grams_temp.append((words[i], filename, index, index))
                    index += 1
                    
                n_grams_temp = []    # the return list
                for i in range(len(words)):
                    temp = words[i]
                    for j in range(1, n):
                        if (i + j) < len(words):
                            temp = temp + ' ' + words[i + j]
                            temp_with_first_index = (temp, filename, single_grams_temp[i][2], single_grams_temp[i + j][2], feature_single_quote_temp[i + j])
                            n_grams_temp.append(temp_with_first_index)

                # single_grams += n_grams
                for i in range(len(single_grams_temp)):
                    n_grams_temp.append(single_grams_temp[i] + (feature_single_quote_temp[i],))

                n_grams.extend(n_grams_temp)
                single_grams.extend(single_grams_temp)
    return n_grams, single_grams, single_grams2

def eliminate_all_lower(ngrams):
    """
    Take out n-gram which does not have any word capitalised
    :param ngrams: all n-grams
    :return: all n-grams for each n-gram has a least one word capitalised
    """
    new_ngram = []
    for ngram in ngrams:
        for word in ngram[0].split(' '):
            if len(word) > 0 and word[0].isupper():
                new_ngram.append(ngram)
                break

    return new_ngram
