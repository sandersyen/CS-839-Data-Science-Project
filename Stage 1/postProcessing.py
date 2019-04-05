def take_out_overlapped(ngrams, predict, label):
    """
    Take out n-gram which is the subset of another n-gram
    :param ngrams: all n-grams
    :return: the remaining n-grams
    """
    new_ngrams, new_predict, new_label, prev, prev_predict = [], [], [], None, 0
    for element_index in range(len(ngrams)):
        # if prev is None || (filenames are different) || (starting index are different)
        if not prev \
            or ngrams[element_index][1] != prev[1] \
            or ngrams[element_index][2] == 0 \
            or prev_predict == 0 \
            or (#ngrams_labels_predicts_sets[element_index][0][1] == prev[0][1] \
                # pre[element_index]==1 \
                # and prev[2]==1 \
                not(prev[2] <= ngrams[element_index][2] <= prev[3]) \
                or not(prev[2] <= ngrams[element_index][3] <= prev[3])):
            prev = ngrams[element_index]
            prev_predict = predict[element_index]
            new_ngrams.append(ngrams[element_index])
            new_predict.append(predict[element_index])
            new_label.append(label[element_index])

    return new_ngrams, new_predict, new_label


def set_predict_value(ngrams, predict):
    for element_index in range(len(ngrams)):
        # 19: start_end_dash, 5: contains_country, 10: contains_prefix, 12: contains_organization, 18: contains_verb,\
        # 6: contains_conjunction, 29: start_with_suffix, 30: contains_common_adj
        if ngrams[element_index][19] == 1 or ngrams[element_index][5] == 1 \
            or ngrams[element_index][29] == 1 or ngrams[element_index][6] == 1 \
            or ngrams[element_index][10] == 1 or ngrams[element_index][12] == 1 \
            or ngrams[element_index][18] == 1 or ngrams[element_index][30] == 1:
            predict[element_index] = 0

    return predict
