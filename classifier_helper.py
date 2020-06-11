#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 22:33:19 2020

classifier_helper.py
helper functions for tweet_classifier.py

@author: charly
"""
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from classifier_plots import ROC_plot

def normalize_X(X_train, X_test):
    '''Normalizes training and test sets'''
    #sparse matrix: with_mean cannot be set to True
    scaler = preprocessing.StandardScaler(with_mean = False)
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm  = scaler.transform(X_test) # use same parameters as in training

    return X_train_norm, X_test_norm


def listify_dict_values(dict):
    '''Turns each value in dictionary into a list for grid search'''
    for key, value in dict.items(): dict[key] = [value]
    return dict


def get_classifiers():
    '''
    returns dictionary of available classifiers
    this list should match the names in get_classifier_names
    '''
    algos = {'LR' : LogisticRegression(),
             'SVM': SVC(),
             'RF' : RandomForestClassifier(),
             'GB' : GradientBoostingClassifier(),
             'MLP': MLPClassifier(),}
    return algos


def get_classifier(name):
    classifiers = get_classifiers()
    try:
        return classifiers[name]
    except:
        print(f'[get_classifier]: invalid classifier name: {name}')
        sys.exit()


def get_classifier_names():
    '''
    returns dictionary of names of available classifiers
    this list should match the names in get_classifiers
    '''
    names = {'LR' : 'Logistic Regression',
             'SVM': 'SVM',
             'RF' : 'Random Forest',
             'GB' :'Boosted Trees',
             'MLP': 'Multilayer Perceptron',}
    return names


def get_classifier_name(name):
    classifiers = get_classifier_names()
    try:
        return classifiers[name]
    except:
        print(f'[get_classifier_name]: invalid classifier name: {name}')
        sys.exit()


def describe_run(clf_flags, method, model_type, randomized, flag):
    '''
    Outputs a description of running algorithms to stdout
    clf_flags:  dictionary of classifier flags eg {'LR': True,...}
    method:     'TFIDF' or 'GOW'
    tuning:     tuning (True) or running (False)
    randomized: True/False - ony applies when tuning is True
    flag:       'start' or 'end' according to beginning or end of run
    '''
    names = get_classifier_names()

    string_ = '*** '
    if flag == 'start': string_ += 'Starting '

    if model_type == 'tuning':
        if randomized == True: string_ += 'randomized '
        else                 : string_ += 'exhaustive '
        string_ += 'tuning '
    else:
        string_ += 'computing '

    string_ += f'{method} / '

    for i, (key, value) in enumerate(clf_flags.items()):
        if value == True: string_ += f'{names[key]} '

    if model_type == 'base':
        string_ += '(base model) '
    elif model_type == 'opt':
        string_ += '(optimized model) '

    if flag == 'end': string_ += ' ended'
    string_ += ' ***\n'
    print(string_)


def ROC(clf, name_alg, method, X_train, X_test, y_train, y_test, save=True):
    '''
    builds & trains OneVsRestClassifier classsifier from existing classifier
    OneVsRestClassifier trains classifier
    computes ROC & AUC curves & sends to plot
    '''
    # Binarize the output
    lb        = preprocessing.LabelBinarizer()
    y_train   = lb.fit_transform(y_train)
    y_test    = lb.fit_transform(y_test)
    n_classes = y_train.shape[1]

    classifier = OneVsRestClassifier(clf)
    if name_alg in ('LR', 'SVM'):
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    else:
        y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

    # Compute ROC & AUC curves
    fpr, tpr, thresholds, ix, gmeans, roc_auc = ({} for i in range(6))
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i]                    = auc(fpr[i], tpr[i])
        gmeans[i]                     = np.sqrt(tpr[i] * (1-fpr[i]))
        ix[i]                         = np.argmax(gmeans[i])
        print(f'{ix[i]:3d}: gmeans={gmeans[i][ix[i]]:.3f} thresholds={thresholds[i][ix[i]]:.3f}')

    mean_gmeans = 0
    for i in range(n_classes):
        idx =  ix[i]
        mean_gmeans += gmeans[i][idx]
    mean_gmeans /= n_classes
    ROC_plot(fpr, tpr, roc_auc, n_classes, ix, mean_gmeans, name_alg, method, save)


def count_grid_combinations(param_grid):
    '''param_grid is either a list of dictionaries or a dictionary'''
    if isinstance(param_grid, list):
        ncombos = 0
        for params in param_grid:
            dic_combos = 1
            for key, values in params.items():
                dic_combos *= len(values)
            ncombos += dic_combos
    elif isinstance(param_grid, dict):
        ncombos = 1
        for key, values in param_grid.items():
            ncombos *= len(values)
    else:
        ValueError(f'[count_combinations] Invalid parameter grid: {param_grid}')
    return ncombos