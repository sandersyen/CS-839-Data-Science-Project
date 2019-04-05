from sklearn.model_selection import cross_val_score, ShuffleSplit
# from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import preprocessing
from ngramGenerator import *
from featureIdentifier import *
from mlModel import *
from postProcessing import *
import pandas as pd
from pandas import DataFrame


def main():
    articles, train_labels_set,  test_labels_set = [], set(), set()

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ''' Pre-processing                                                   '''
    ''' (1) Load data and split data into train/test sets                '''
    ''' (2) Hashset the labels and remove labels on the data             '''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # add all files' data into articles
    preprocessing.read_data(articles)

    # split data to train and test sets
    train_set, test_set = preprocessing.data_split(articles)
    train_label_count, test_label_count = 0, 0

    # take off label and add names to labels
    for i in range(len(train_set)):
        train_set[i], train_label_count, train_labels_set =\
            preprocessing.label_extraction_takeoff(paragraphs=train_set[i], count=train_label_count, labels=train_labels_set)

    for i in range(len(test_set)):
        test_set[i], test_label_count, test_labels_set =\
            preprocessing.label_extraction_takeoff(paragraphs=test_set[i], count=test_label_count, labels=test_labels_set)

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ''' N-gram generation                                                '''
    ''' (1) Generate all n-gram (with first feature whether contains 's) '''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    train_ngram_result, test_ngram_result = [], []
    train_single_gram, test_single_gram = [], []
    train_single_gram2, test_single_gram2 = [], []        # save single ones in order for later use

    for i in range(len(train_set)):
        ngrams, singles, singles2 = generate_ngrams(filename=train_set[i][0], content=train_set[i][1], n=5)
        train_ngram_result.append(ngrams)
        train_single_gram.append(singles)
        train_single_gram2.append(singles2)

    for i in range(len(test_set)):
        ngrams, singles, singles2 = generate_ngrams(filename=test_set[i][0], content=test_set[i][1], n=5)
        test_ngram_result.append(ngrams)
        test_single_gram.append(singles)
        test_single_gram2.append(singles2)

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ''' Take out n-gram with only lowercase (only for training data)     '''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    for index in range(len(train_ngram_result)):
        train_ngram_result[index] = eliminate_all_lower(train_ngram_result[index])

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ''' Create a test ngram result without n-gram has only lowercase     '''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    test_ngram_result_without_all_lower = test_ngram_result[:]
    for index in range(len(test_ngram_result_without_all_lower)):
        test_ngram_result_without_all_lower[index] = eliminate_all_lower(test_ngram_result_without_all_lower[index])

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ''' Feature creation                                                 '''
    ''' (1) 's (added during generation of ngram)                        '''
    ''' (2) contains country                                             '''
    ''' (3) contains conjunction                                         '''
    ''' (4) all capitalised                                              '''
    ''' (5) prefix before n-gram                                         '''
    ''' (6) verbs for humans                                             '''
    ''' (7) prefix in n-gram                                             '''
    ''' (8) after preposition                                            '''
    ''' (9) contains organization                                        '''
    ''' (10) has no more than 1 word without capitalised starting letter '''
    ''' (11) contains month                                              '''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    country_set, conjunction_set, prefix_set, verb_set, preposition_set, organ_set, month_set, who_set, common_name_set, common_adj_set = \
        load_country_file(), load_conjunction_file(), load_prefix_library(),\
        load_verb_file(), load_preposition_file(), load_organ_library(), load_month_file(), load_who_file(), load_common_name_file(), load_common_adj_file()

    for ngram_set_index in range(len(train_ngram_result)):
        article = ' '.join(a[0] for a in train_single_gram[ngram_set_index])
        for ngram_index in range(len(train_ngram_result[ngram_set_index])):
            ngram = train_ngram_result[ngram_set_index][ngram_index]

            train_ngram_result[ngram_set_index][ngram_index] = ngram +\
                (contains_country(ngram=ngram, country_set=country_set),
                 contains_conjunction(ngram=ngram, conjunctions_set=conjunction_set),
                 is_all_upper(ngram=ngram),
                 has_prefix_before_ngram(ngram=ngram, single_grams=train_single_gram[ngram_set_index], prefix_set=prefix_set),
                 has_human_verb(ngram=ngram, single_grams=train_single_gram[ngram_set_index], verb_set=verb_set),
                 contains_prefix(ngram=ngram, prefix_set=prefix_set),
                 afterpreposition(ngram=ngram, single_grams=train_single_gram[ngram_set_index], preposition_set=preposition_set),
                 contains_organization(ngram=ngram, organ_set=organ_set),
                 contains_common_name(ngram=ngram, common_name_set=common_name_set),
                 has_duplicate(ngram=ngram),
                 count_occurrences(ngram=ngram, single_grams=article),
                 no_more_than_one_lower(ngram=ngram),
                 contains_month(ngram=ngram, month_set=month_set),
                 contains_verb(ngram=ngram, verb_set=verb_set),
                 start_end_dash(ngram=ngram),
                 all_upper_character(ngram=ngram),
                 word_length(ngram=ngram),
                 has_fullstop_before_ngram(ngram=ngram, single_grams2=train_single_gram2[ngram_set_index]),
                 has_comma_before_ngram(ngram=ngram, single_grams2=train_single_gram2[ngram_set_index]),
                 before_who(ngram=ngram, single_grams=train_single_gram[ngram_set_index], who_set=who_set),
                 has_comma(ngram=ngram, single_grams2=train_single_gram2[ngram_set_index]),
                 has_who(ngram=ngram, who_set=who_set),
                 is_name_suffix(ngram=ngram),
                 has_one_dash(ngram=ngram),
                 start_with_suffix(ngram=ngram),
                 contains_common_adj(ngram=ngram, common_adj_set=common_adj_set),)

    for ngram_set_index in range(len(test_ngram_result_without_all_lower)):
        article = ' '.join(a[0] for a in test_single_gram[ngram_set_index])
        for ngram_index in range(len(test_ngram_result_without_all_lower[ngram_set_index])):
            ngram = test_ngram_result_without_all_lower[ngram_set_index][ngram_index]
            test_ngram_result_without_all_lower[ngram_set_index][ngram_index] = ngram +\
                (contains_country(ngram=ngram, country_set=country_set),
                 contains_conjunction(ngram=ngram, conjunctions_set=conjunction_set),
                 is_all_upper(ngram=ngram),
                 has_prefix_before_ngram(ngram=ngram, single_grams=test_single_gram[ngram_set_index], prefix_set=prefix_set),
                 has_human_verb(ngram=ngram, single_grams=test_single_gram[ngram_set_index], verb_set=verb_set),
                 contains_prefix(ngram=ngram, prefix_set=prefix_set),
                 afterpreposition(ngram=ngram, single_grams=test_single_gram[ngram_set_index], preposition_set=preposition_set),
                 contains_organization(ngram=ngram, organ_set=organ_set),
                 contains_common_name(ngram=ngram, common_name_set=common_name_set),
                 has_duplicate(ngram=ngram),
                 count_occurrences(ngram=ngram, single_grams=article),
                 no_more_than_one_lower(ngram=ngram),
                 contains_month(ngram=ngram, month_set=month_set),
                 contains_verb(ngram=ngram, verb_set=verb_set),
                 start_end_dash(ngram=ngram),
                 all_upper_character(ngram=ngram),
                 word_length(ngram=ngram),
                 has_fullstop_before_ngram(ngram=ngram, single_grams2=test_single_gram2[ngram_set_index]),
                 has_comma_before_ngram(ngram=ngram, single_grams2=test_single_gram2[ngram_set_index]),
                 before_who(ngram=ngram, single_grams=test_single_gram[ngram_set_index], who_set=who_set),
                 has_comma(ngram=ngram, single_grams2=test_single_gram2[ngram_set_index]),
                 has_who(ngram=ngram, who_set=who_set),
                 is_name_suffix(ngram=ngram),
                 has_one_dash(ngram=ngram),
                 start_with_suffix(ngram=ngram),
                 contains_common_adj(ngram=ngram, common_adj_set=common_adj_set),)

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ''' Train DT, SVM, NB                                                '''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    train_ngrams = []
    while len(train_ngram_result):
        train_ngrams.extend(train_ngram_result.pop())
    train_ngrams = sorted(train_ngrams, key=lambda i: (int(i[1]), i[2], i[3]-i[2]), reverse=True)
    new_train, train_label = features_label_separator(ngrams=train_ngrams, labels_set=train_labels_set)

    decision_tree = build_decision_tree(data=new_train, label=train_label)
    support_vector_machine = build_support_vector_machine(data=new_train, label=train_label)
    nb_classifier = build_nb_classifier(data=new_train, label=train_label)
    rf_classifier = build_rf_classifier(data=new_train, label=train_label)
    lr_classifier = build_lr_classifier(data=new_train, label=train_label)

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ''' merge test ngram result                                          '''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    test_ngrams = []
    while len(test_ngram_result_without_all_lower):
        test_ngrams.extend(test_ngram_result_without_all_lower.pop())
    test_ngrams = sorted(test_ngrams, key=lambda i: (int(i[1]), i[2], i[3]-i[2]), reverse=True)
    new_test, test_label = features_label_separator(ngrams=test_ngrams, labels_set=test_labels_set)
    
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ''' use DT, SVM, NB, RF, LR to predict test set                      '''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    print("''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
    print("Train Set")
    print("''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
    print(" ")
    print("Number of Name: ")
    print(train_label_count)
    decision_tree_predict_train = decision_tree.predict(new_train)
    support_vector_machine_predict_train = support_vector_machine.predict(new_train)
    nb_classifier_predict_train = nb_classifier.predict(new_train)
    rf_classifier_predict_train = rf_classifier.predict(new_train)
    lr_classifier_predict_train = lr_classifier.predict(new_train)

    print("precision before post processing:")
    print 'lr: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(lr_classifier_predict_train, train_label)])) / sum(lr_classifier_predict_train))
    print 'dt: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(decision_tree_predict_train, train_label)])) / sum(decision_tree_predict_train))  
    print 'svm: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(support_vector_machine_predict_train, train_label)])) / sum(support_vector_machine_predict_train))  
    print 'nb: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(nb_classifier_predict_train, train_label)])) / sum(nb_classifier_predict_train))  
    print 'rf: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(rf_classifier_predict_train, train_label)])) / sum(rf_classifier_predict_train))  
    print('')
    print("recall before post processing:")
    print 'lr: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(lr_classifier_predict_train, train_label)])) / sum(train_label))
    print 'dt: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(decision_tree_predict_train, train_label)])) / sum(train_label))  
    print 'svm: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(support_vector_machine_predict_train, train_label)])) / sum(train_label))  
    print 'nb: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(nb_classifier_predict_train, train_label)])) / sum(train_label))  
    print 'rf: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(rf_classifier_predict_train, train_label)])) / sum(train_label))  
    print('')
    decision_tree_ngrams_train, decision_tree_predict_train, decision_tree_label_train = take_out_overlapped(train_ngrams, decision_tree_predict_train, train_label)
    support_vector_machine_ngrams_train, support_vector_machine_predict_train, support_vector_machine_label_train = take_out_overlapped(train_ngrams, support_vector_machine_predict_train, train_label)
    nb_classifier_ngrams_train, nb_classifier_predict_train, nb_classifier_label_train = take_out_overlapped(train_ngrams, nb_classifier_predict_train, train_label)
    rf_classifier_ngrams_train, rf_classifier_predict_train, rf_classifier_label_train = take_out_overlapped(train_ngrams, rf_classifier_predict_train, train_label)
    lr_classifier_ngrams_train, lr_classifier_predict_train, lr_classifier_label_train = take_out_overlapped(train_ngrams, lr_classifier_predict_train, train_label)

    decision_tree_predict_train = set_predict_value(ngrams=decision_tree_ngrams_train, predict=decision_tree_predict_train)
    support_vector_machine_predict_train = set_predict_value(ngrams=support_vector_machine_ngrams_train, predict=support_vector_machine_predict_train)
    nb_classifier_predict_train = set_predict_value(ngrams=nb_classifier_ngrams_train, predict=nb_classifier_predict_train)
    rf_classifier_predict_train = set_predict_value(ngrams=rf_classifier_ngrams_train, predict=rf_classifier_predict_train)
    lr_classifier_predict_train = set_predict_value(ngrams=lr_classifier_ngrams_train, predict=lr_classifier_predict_train)

    print("precision:")
    print 'lr: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(lr_classifier_predict_train, lr_classifier_label_train)])) / sum(lr_classifier_predict_train))
    print 'dt: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(decision_tree_predict_train, decision_tree_label_train)])) / sum(decision_tree_predict_train))
    print 'svm: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(support_vector_machine_predict_train, support_vector_machine_label_train)])) / sum(support_vector_machine_predict_train))
    print 'nb: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(nb_classifier_predict_train, nb_classifier_label_train)])) / sum(nb_classifier_predict_train))
    print 'rf: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(rf_classifier_predict_train, rf_classifier_label_train)])) / sum(rf_classifier_predict_train))
    print('')
    print("recall:")
    print 'lr: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(lr_classifier_predict_train, lr_classifier_label_train)])) / sum(lr_classifier_label_train))
    print 'dt: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(decision_tree_predict_train, decision_tree_label_train)])) / sum(decision_tree_label_train))
    print 'svm: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(support_vector_machine_predict_train, support_vector_machine_label_train)])) / sum(support_vector_machine_label_train))
    print 'nb: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(nb_classifier_predict_train, nb_classifier_label_train)])) / sum(nb_classifier_label_train))
    print 'rf: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(rf_classifier_predict_train, rf_classifier_label_train)])) / sum(rf_classifier_label_train))

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ''' use DT, SVM, NB, RF, LR to predict test set                      '''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    print("''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
    print("Test Set")
    print("''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
    print(" ")
    print("Number of Name: ")
    print(test_label_count)
    decision_tree_predict = decision_tree.predict(new_test)
    support_vector_machine_predict = support_vector_machine.predict(new_test)
    nb_classifier_predict = nb_classifier.predict(new_test)
    rf_classifier_predict = rf_classifier.predict(new_test)
    lr_classifier_predict = lr_classifier.predict(new_test)

    print("precision before post processing:")
    print 'lr: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(lr_classifier_predict, test_label)])) / sum(lr_classifier_predict))
    print 'dt: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(decision_tree_predict, test_label)])) / sum(decision_tree_predict))  
    print 'svm: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(support_vector_machine_predict, test_label)])) / sum(support_vector_machine_predict))  
    print 'nb: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(nb_classifier_predict, test_label)])) / sum(nb_classifier_predict))  
    print 'rf: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(rf_classifier_predict, test_label)])) / sum(rf_classifier_predict))  
    print('')
    print("recall before post processing:")
    print 'lr: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(lr_classifier_predict, test_label)])) / sum(test_label))
    print 'dt: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(decision_tree_predict, test_label)])) / sum(test_label))  
    print 'svm: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(support_vector_machine_predict, test_label)])) / sum(test_label))  
    print 'nb: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(nb_classifier_predict, test_label)])) / sum(test_label))  
    print 'rf: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(rf_classifier_predict, test_label)])) / sum(test_label))  
    print('')
    decision_tree_ngrams, decision_tree_predict, decision_tree_label = take_out_overlapped(test_ngrams, decision_tree_predict, test_label)
    support_vector_machine_ngrams, support_vector_machine_predict, support_vector_machine_label = take_out_overlapped(test_ngrams, support_vector_machine_predict, test_label)
    nb_classifier_ngrams, nb_classifier_predict, nb_classifier_label = take_out_overlapped(test_ngrams, nb_classifier_predict, test_label)
    rf_classifier_ngrams, rf_classifier_predict, rf_classifier_label = take_out_overlapped(test_ngrams, rf_classifier_predict, test_label)
    lr_classifier_ngrams, lr_classifier_predict, lr_classifier_label = take_out_overlapped(test_ngrams, lr_classifier_predict, test_label)

    decision_tree_predict = set_predict_value(ngrams=decision_tree_ngrams, predict=decision_tree_predict)
    support_vector_machine_predict = set_predict_value(ngrams=support_vector_machine_ngrams, predict=support_vector_machine_predict)
    nb_classifier_predict = set_predict_value(ngrams=nb_classifier_ngrams, predict=nb_classifier_predict)
    rf_classifier_predict = set_predict_value(ngrams=rf_classifier_ngrams, predict=rf_classifier_predict)
    lr_classifier_predict = set_predict_value(ngrams=lr_classifier_ngrams, predict=lr_classifier_predict)

    print("precision:")
    print 'lr: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(lr_classifier_predict, lr_classifier_label)])) / sum(lr_classifier_predict))
    print 'dt: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(decision_tree_predict, decision_tree_label)])) / sum(decision_tree_predict))
    print 'svm: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(support_vector_machine_predict, support_vector_machine_label)])) / sum(support_vector_machine_predict))
    print 'nb: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(nb_classifier_predict, nb_classifier_label)])) / sum(nb_classifier_predict))
    print 'rf: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(rf_classifier_predict, rf_classifier_label)])) / sum(rf_classifier_predict))
    print('')
    print("recall:")
    print 'lr: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(lr_classifier_predict, lr_classifier_label)])) / sum(lr_classifier_label))
    print 'dt: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(decision_tree_predict, decision_tree_label)])) / sum(decision_tree_label))
    print 'svm: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(support_vector_machine_predict, support_vector_machine_label)])) / sum(support_vector_machine_label))
    print 'nb: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(nb_classifier_predict, nb_classifier_label)])) / sum(nb_classifier_label))
    print 'rf: ' + str(float(sum([1 if a == b and a == 1 else 0 for a, b in zip(rf_classifier_predict, rf_classifier_label)])) / sum(rf_classifier_label))
    
    # print ("==========================================================================")
    # print("data frame:")

    # df = pd.DataFrame(columns=['words', 'predict', 'label'])
    # for i in range(len(rf_classifier_predict)):
    #     if not (rf_classifier_predict[i] == rf_classifier_label[i])  and rf_classifier_predict[i] == 1:
    #         df = df.append({'words': rf_classifier_ngrams[i], 'predict': rf_classifier_predict[i], 'label':rf_classifier_label[i]}, ignore_index = True)
    # DataFrame.to_csv(df, "rf_classifier_predict.csv", index=False)
    # scores = cross_val_score(svm.SVC(), new_train, train_label, cv=ShuffleSplit(n_splits=5, test_size=0.3, random_state=0))
    # print (scores)


if __name__ == "__main__":
    main()
