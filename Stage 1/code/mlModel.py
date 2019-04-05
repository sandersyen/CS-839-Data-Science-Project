from sklearn.linear_model import LogisticRegression
from sklearn import tree, svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


def build_decision_tree(data, label):
    """
    Build the decision tree based on the data and its corresponding label
    :param data: a list of tuple which contains all features of a data
    :param label: a list of label for data
    :return: a trained decision tree
    """
    dt_tree = tree.DecisionTreeClassifier()
    return dt_tree.fit(data, label)


def build_support_vector_machine(data, label):
    """
    Build the support vector machine based on the data and its corresponding label
    :param data: a list of tuple which contains all features of a data
    :param label: a list of label for data
    :return: trained support vector machine
    """
    trained_svm = svm.SVC(gamma='scale', C=100)
    return trained_svm.fit(data, label)


def build_nb_classifier(data, label):
    """
    Build the naive bayes classifier based on the data and its corresponding label
    :param data: a list of tuple which contains all features of a data
    :param label: a list of label for data
    :return: trained naive bayes classifier
    """
    classifier = BernoulliNB()
    return classifier.fit(data, label)


def build_rf_classifier(data, label):
    """
    Build the random forest classifier based on the data and its corresponding label
    :param data: a list of tuple which contains all features of a data
    :param label: a list of label for data
    :return: trained naive bayes classifier
    """
    # pipe = make_pipeline(StandardScaler(),RandomForestClassifier())
    # param_grid = {'n_estimators': list(range(1, 30))}
    # gs = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, \
    #     iid=False, n_jobs=-1, refit=True,scoring='accuracy',cv=10)
    # gs.fit(data, label)
    # n_estimators=gs.best_params_['n_estimators']
    classifier = RandomForestClassifier(n_estimators=34, n_jobs=-1, criterion='gini', class_weight={0: 1, 1: 1.45}, random_state=10)
    return classifier.fit(data, label)


def build_lr_classifier(data, label):
    """
    Build the logistic regression classifier based on the data and its corresponding label
    :param data: a list of tuple which contains all features of a data
    :param label: a list of label for data
    :return: trained logistic regression classifier
    """
    classifier = LogisticRegression(solver='newton-cg',n_jobs=-1,class_weight={0: 1, 1: 1.5})
    return classifier.fit(data, label)
