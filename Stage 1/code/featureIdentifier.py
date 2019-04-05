def load_who_file():
    """
    Read file whos.txt and insert data to a hash set
    :return: the hash set with all whos from file whos.txt
    """
    return set([who.strip('\n').lower() for who in file("data/whos.txt", 'r').readlines()])

def load_common_name_file():
    """
    Read file common_name.txt and insert data to a hash set
    :return: the hash set with all whos from file common_name.txt
    """
    return set([common_name.strip('\n').lower() for common_name in file("data/common_name.txt", 'r').readlines()])

def load_common_adj_file():
    """
    Read file common_adj.txt and insert data to a hash set
    :return: the hash set with all whos from file common_adj.txt
    """
    return set([common_adj.strip('\n').lower() for common_adj in file("data/common_adj.txt", 'r').readlines()])

def load_country_file():
    """
    Read file countries.txt and insert data to a hash set
    :return: the hash set with all countries name from file countries.txt
    """
    return set([country.strip('\n').lower() for country in file("data/countries.txt", 'r').readlines()])

def load_conjunction_file():
    """
    Read file conjunctions.txt and insert data to a hash set
    :return: the hash set with all conjunctions from file conjunctions.txt
    """
    return set([conjunctions.strip('\n').lower() for conjunctions in file("data/conjunctions.txt", 'r').readlines()])

def load_prefix_library():
    """
    Generate a hash set with all prefixes
    :return: the hash set with all prefixes
    """
    return set([prefix.strip('\n').lower() for prefix in file("data/prefix.txt", 'r').readlines()])

def load_organ_library():
    """
    Generate a hash set with all organization titles
    :return: the hash set with all organization titles
    """
    return set([organ.strip('\n').lower() for organ in file("data/organization.txt", 'r').readlines()])

def load_month_file():
    """
    Generate a hash set with all months
    :return: the hash set with all months
    """
    return set([month.strip('\n').lower() for month in file("data/month.txt", 'r').readlines()])

def load_verb_file():
    """
    Read files irregular_verbs.txt and regular_verbs.txt and insert data to a hash set
    :return: the hash set with all verbs from files
    """
    return set(file("data/irregular_verbs.txt", 'r').read().split(', ')) |\
           set(file("data/regular_verbs.txt", 'r').read().split(', '))

def load_preposition_file():
    """
    Read files preposition.txt and insert data to a hash set
    :return: the hash set with all preposition from files
    """
    return set(open("data/preposition.txt", 'r').read().split(', '))

def contains_country(ngram, country_set):
    """
    Identify if a n-gram has countries, return 1 (has feature) 0 (no such feature)
    :param ngram: a ngram
    :param country_set: a set contains all countries
    :return: 1 (has feature) or 0 (no such feature)
    """
    # find if any word in current ngram has country name
    words = ngram[0].split(' ')
    for word in words:
        if word.lower() in country_set:
            return 1
    if len(words) >= 2:
        for i in range(1, len(words)):
            if (words[i-1]+' '+words[i]).lower() in country_set:
                return 1
    if len(words) >= 3:
        for i in range(2, len(words)):
            if (words[i-2]+' '+words[i-1]+' '+words[i]).lower() in country_set:
                return 1
    return 0


def contains_common_name(ngram, common_name_set):
    """
    Identify if a n-gram has common name, return 1 (has feature) 0 (no such feature)
    :param ngram: a ngram
    :param common_name_set: a set contains all common name
    :return: 1 (has feature) or 0 (no such feature)
    """
    # find if any word in current ngram has common name
    for word in ngram[0].split(' '):
        if word.lower() in common_name_set:
            return 1
    return 0

def contains_common_adj(ngram, common_adj_set):
    """
    Identify if a n-gram has common adj, return 1 (has feature) 0 (no such feature)
    :param ngram: a ngram
    :param common_name_set: a set contains all common adj
    :return: 1 (has feature) or 0 (no such feature)
    """
    # find if any word in current ngram has common adj
    for word in ngram[0].split(' '):
        if word.lower() in common_adj_set:
            return 1
    return 0

def contains_prefix(ngram, prefix_set):
    """
    Identify if a n-gram has prefix, return 1 (has feature) 0 (no such feature)
    :param ngram: a ngram
    :param prefix_set: a set contains all prefixes
    :return: 1 (has feature) or 0 (no such feature)
    """
    # find if any word in current ngram has country name
    for word in ngram[0].split(' '):
        if word.lower() in prefix_set:
            return 1
    return 0

def contains_month(ngram, month_set):
    """
    Identify if a n-gram has month, return 1 (has feature) 0 (no such feature)
    :param ngram: a ngram
    :param month_set: a set contains all months
    :return: 1 (has feature) or 0 (no such feature)
    """
    # find if any word in current ngram has country name
    for word in ngram[0].split(' '):
        if word.lower() in month_set:
            return 1
    return 0

def contains_organization(ngram, organ_set):
    """
    Identify if a n-gram has organization titles, return 1 (has feature) 0 (no such feature)
    :param ngram: a ngram
    :param organ_set: a set contains commom organization titles
    :return: 1 (has feature) or 0 (no such feature)
    """
    # find if any word in current ngram has country name
    words = ngram[0].split(' ')
    for word in words:
        if word.lower() in organ_set:
            return 1

    if len(words) >= 2:
        for i in range(1, len(words)):
            if (words[i-1]+' '+words[i]).lower() in organ_set:
                return 1
    return 0

def contains_conjunction(ngram, conjunctions_set):
    """
    Identify if a n-gram has conjunctions, return 1 (has feature) 0 (no such feature)
    :param ngram: a ngram
    :param conjunctions_set: a set contains all conjunctions
    :return: 1 (has feature) 0 (no such feature)
    """
    # find if any word in current ngram has country name
    for word in ngram[0].split(' '):
        if word.lower() in conjunctions_set:
            return 1
    return 0

def contains_verb(ngram, verb_set):
    """
    Identify if a n-gram has conjunctions, return 1 (has feature) 0 (no such feature)
    :param ngram: a ngram
    :param conjunctions_set: a set contains all conjunctions
    :return: 1 (has feature) 0 (no such feature)
    """
    # find if any word in current ngram has country name
    for word in ngram[0].split(' '):
        if word.lower() in verb_set:
            return 1
    return 0

def is_all_upper(ngram):
    """
    Check all the words in the content if all words start with upper case
    :param ngram: a ngram
    :return: 1 (has feature) 0 (no such feature)
    """
    for word in ngram[0].split(' '):
        if len(word) > 0 and word[0].islower():
            return 0
    return 1

def has_who(ngram, who_set):
    """
    Check all the words in the content if it has who
    :param ngram: a ngram
    :return: 1 (has feature) 0 (no such feature)
    """
    for word in ngram[0].split(' '):
        if word.lower() in who_set:
            return 1
    return 0

def no_more_than_one_lower(ngram):
    """
    Check all the words in the content if all words has less than 2 lower case at each starting letter
    :param ngram: a ngram
    :return: 1 (has feature) 0 (no such feature)
    """
    count = 0
    for word in ngram[0].split(' '):
        if word.islower():
            count += 1
            if count > 1:
                return 0
    return 1

def has_prefix_before_ngram(ngram, single_grams, prefix_set):
    """
    Check if the word in front of the input ngram is a prefix for name
    :param ngram: a n-gram
    :param single_grams: all words in an article with order
    :param prefix_set: a set contains all prefixes
    :return: 1 (has feature) 0 (no such feature)
    """
    if (ngram[2] - 1) >= 0:
        preWord = single_grams[ngram[2] - 1][0].lower()
        if preWord in prefix_set:
            return 1
    return 0

def has_human_verb(ngram, single_grams, verb_set):
    """
    Check if the word after the input ngram is a verb usually used for human
    :param ngram: a n-gram
    :param single_grams: all words in an article with order
    :param verb_set: a set contains all verbs usually used for human
    :return: 1 (has feature) 0 (no such feature)
    """
    ngram_end_index = ngram[3]
    if (ngram_end_index + 1) < len(single_grams):
        if single_grams[ngram_end_index+1][0] in verb_set:
            return 1
    return 0

def features_label_separator(ngrams, labels_set=None):
    """
    Separate features and label from n-grams and return two lists
    :param ngrams: all n-grams from all articles
    :param labels_set: the hash set of all labels
    :return: two lists -- features and label from n-grams
    """
    features = [ngram[4:] for ngram in ngrams]
    label = [1 if ngram[0] in labels_set else 0 for ngram in ngrams] if labels_set else []
    return features, label

def afterpreposition(ngram, single_grams, preposition_set):
    """
    Check if the word in front of the input ngram is a preposition
    :param ngram: a n-gram
    :param single_grams: all words in an article with order
    :param preposition_set: a set contains all prefixes
    :return: 1 (has feature) 0 (no such feature)
    """
    if (ngram[2] - 1) >= 0:
        prepos = single_grams[ngram[2] - 1][0].lower()
        if prepos in preposition_set:
            return 1
    return 0

def before_who(ngram, single_grams, who_set):
    """
    Check if the word after the input ngram is "who"
    :param ngram: a n-gram
    :param single_grams: all words in an article with order
    :param who_set: a set contains who
    :return: 1 (has feature) 0 (no such feature)
    """
    return 1 if (ngram[2]+1) < len(single_grams) and (single_grams[ngram[2]+1][0]).lower() in who_set else 0

def has_duplicate(ngram):
    """
    Check if the words in input ngram has any duplicate words
    :param ngram: a n-gram
    :return: 1 (has feature) 0 (no such feature)
    """
    words = set()
    for word in ngram[0].split(' '):
        if word in words:
            return 1
        else:
            words.add(word)
    return 0

def count_occurrences(ngram, single_grams):
    """
    Count the word's occurrences in the article(only for single words)
    :param ngram: a n-gram
    :param single_grams: all words in an article with order
    :return: word's occurrences
    """
    # print (single_grams)
    # data = ' '.join(a[0] for a in single_grams)
    return single_grams.count(ngram[0])

def start_end_dash(ngram):
    """
    Check if the ngram contain string starts or ends with dash
    :param ngram: a n-gram
    :return: 1 (has feature) 0 (no such feature)
    """
    words = ngram[0].split(' ')
    if not words[0].isalpha() or (len(words) > 1 and not words[-1].isalpha()) or words.count('-') > 1:
        return 1        
    count = 0
    for word in words:
        count += word.count('-')
    if count > 1:
        return 1
    return 0

def has_one_dash(ngram):
    """
    Check if the ngram contain exactly one dash
    :param ngram: a n-gram
    :return: 1 (has feature) 0 (no such feature)
    """
    words = ngram[0].split(' ')
    if words.count('-') == 1:
        return 1        
    count = 0
    for word in words:
        count += word.count('-')
    if count == 1:
        return 1
    return 0

def all_upper_character(ngram):
    """
    Check all the words in the content if all character in words is upper case
    :param ngram: a ngram
    :return: 1 (has feature) 0 (no such feature)
    """
    for word in ngram[0].split(' '):
        if word.isupper():
            return 1
    return 0

def word_length(ngram):
    """
    Check number of words
    :param ngram: a ngram
    :return: number of words
    """
    words = ngram[0].split(' ')
    return len(words)

def has_fullstop_before_ngram(ngram, single_grams2):
    """
    Check if the word in front of the input ngram is a fullstop
    :param ngram: a n-gram
    :param single_grams2: all words including punctuation in an article with order
    :return: 1 (has feature) 0 (no such feature)
    """
    if (ngram[2] - 1) >= 0:
        preWord = single_grams2[ngram[2] - 1][0].lower()
        if preWord.endswith("."):
            return 1
    return 0

def has_comma_before_ngram(ngram, single_grams2):
    """
    Check if the word in front of the input ngram is a comma
    :param ngram: a n-gram
    :param single_grams2: all words including punctuation in an article with order
    :return: 1 (has feature) 0 (no such feature)
    """
    if (ngram[2] - 1) >= 0:
        preWord = single_grams2[ngram[2] - 1][0].lower()
        if preWord.endswith(","):
            return 1
    return 0

def has_comma(ngram, single_grams2):
    """
    Check if the word has a comma in the end
    :param ngram: a n-gram
    :param single_grams2: all words including punctuation in an article with order
    :return: 1 (has feature) 0 (no such feature)
    """
    lastWord = single_grams2[ngram[3]][0]
    if lastWord.endswith(","):
        return 1
    return 0

def is_name_suffix(ngram):
    """
    Check if the word has a suffix
    :param ngram: a n-gram
    :return: 1 (has feature) 0 (no such feature)
    """
    suffixes = ['Sr', 'Sr.', 'Jr', 'Jr.', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX',
                'sr', 'sr.', 'jr', 'jr.', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi', 'xii', 'xiii', 'xiv', 'xv', 'xvi', 'xvii', 'xviii', 'xix', 'xx']
    for word in ngram[0].split(' '):
        if word in suffixes:
            return 1
    return 0

def start_with_suffix(ngram):
    """
    Check if the word has a suffix
    :param ngram: a n-gram
    :return: 1 (has feature) 0 (no such feature)
    """
    suffixes = ['Sr', 'Sr.', 'Jr', 'Jr.', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX',
                'sr', 'sr.', 'jr', 'jr.', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi', 'xii', 'xiii', 'xiv', 'xv', 'xvi', 'xvii', 'xviii', 'xix', 'xx']
    word = ngram[0].split(' ')
    if word[0] in suffixes:
        return 1
    return 0 