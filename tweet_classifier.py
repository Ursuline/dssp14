#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 21:41:54 2020

tweet_classifier.py

TF-IDF & GOW tweet classifier

Algorithms implemented:
- Logistic regression
- SVM
- Random Forest
- Gradient Booster
- MLP

@author: charles mégnin
"""
print(__doc__)

import time
import numpy as np
from pprint import pprint

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from tweet_processor import preprocess
from handles import get_handles
from tweet_io import load_data_from_file, store_model, load_model
from tweet_io import load_processed_data, store_processed_data
from graph_of_words import gow
from tfidf import tf_idf
from classifier_helper import describe_run, get_classifiers, get_classifier_names, normalize_X
from classifier_helper import ROC, listify_dict_values, count_grid_combinations
from classifier_plots import plot_confusion_matrix

def hyper_parameter_tuning(classifier, grid, X_train, y_train, random):
    """
    All-purpose hyper-parameter tuning dispatcher. Expects:
    - A classifier: classifier (e.g. MLPClassifier())
    - A parameter grid: grid
    - The training features and classes: X_train, y_train
    - A boolean variable random to select between:
        * RandomizedSearchCV() -> random=True
        * GridSearchCV() -> random=False
    """
    print(f'[hyper_parameter_tuning] Tuning grid:')
    pprint(grid)
    print(f'[hyper_parameter_tuning] {count_grid_combinations(grid)} possible combinations\n')
    n_jobs  = 10
    n_folds = 3
    n_iter  = 150

    if random:
        print('Random search:')
        clf = RandomizedSearchCV(estimator           = classifier,
                                 param_distributions = grid,
                                 n_iter              = n_iter,
                                 cv                  = n_folds,
                                 n_jobs              = n_jobs,
                                 verbose             = VERBOSE)
    else:
        print('Grid search:')
        clf = GridSearchCV(estimator  = classifier,
                           param_grid = grid,
                           cv         = n_folds,
                           n_jobs     = n_jobs,
                           verbose    = VERBOSE)
    clf.fit(X_train, y_train)
    best_hyper_parameter_summary(clf)
    return clf


def best_hyper_parameter_summary(clf):
    '''Outputs best hyper_parameters from RandomizedSearchCV or GridSearchCV to stdout'''
    print("\nBest parameters set found on development set:")
    pprint(clf.best_params_)

    print("Grid scores on development set:\n")
    means = clf.cv_results_['mean_test_score']
    stds  = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print(f'{mean:.3f} (+/-{std*2:.3f}) -> {params}')

    print("Best estimator:")
    pprint(clf.best_estimator_)

    print("Classification report: (Scores computed on the evaluation set)\n")
    #y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_test, clf.predict(X_test)))
    print(f'Best score grid search: {clf.score(X_train, y_train)}')


def run_algorithm(key, method, model_type, randomized, X_train, y_train, X_test, y_test):
    '''Main dispatcher: sends to either base classifier, tuning, or parameter optimized classifier'''
    clf_algo = clf_algos[key]
    clf_name = clf_names[key]

    if model_type == 'base': # compute base model (no tuning)
        params = set_common_parameters()
        clf    = classify(X_train, y_train, X_test, y_test, params, clf_algo, clf_name, method)
        store_model(clf, key, method, model_type, True) # Save model to file
        print('Parameters currently in use:\n')
        pprint(clf.get_params())

    elif model_type == 'tuning': # Hyper-parameter tuning by exhaustive or random grid search
        if randomized == True: param_grid = build_RandomizedSearch_grid(key)
        else                 : param_grid = build_GridSearch_grid(key, method, model_type)
        clf = hyper_parameter_tuning(clf_algo, param_grid, X_train, y_train, randomized)
        store_model(clf, key, method, model_type, randomized) # Save model to file

    elif model_type == 'opt': # Perform classification with optimized parameters
        if manual_opt: params = set_opt_parameters() # manually load tuned parameters
        else: params = load_opt_parameters(key, method, randomized) # load saved parameters
        params = listify_dict_values(params)
        classify(X_train, y_train, X_test, y_test, params, clf_algo, clf_name, method)

    else:
        raise ValueError(f'[run_algorithm] Invalid model_type: {model_type}')


def result_dispatcher(clf, X_test, y_test, handles, algo, method):
    """
    Dispatches results from fit
    1. Sends fitted classifier to confusion matrix and
    2. outputs results to stdout
    """
    pprint(clf.get_params())
    y_pred = clf.predict(X_test)
    score  = clf.score(X_test, y_test)

    plot_confusion_matrix(y_pred, y_test, score, handles, algo, method, normalize=True)
    print(f'[result_dispatcher] Classification report for {algo} / {method}:\n')
    print(classification_report(y_test, y_pred))
    print(f'{algo} {method} {model_type}: score={score}')


def train_classifier(classifier, params, X_train, y_train):
    '''
    builds classsifier from classifier & its parameters params
    train classifier with X_train & y_train
    returns trained classifier
    '''
    print(params)
    for k, v in params.items():
        if v == None:
            pass
        elif isinstance(v, float) or isinstance(v, int) or isinstance(v, str):
            clf = classifier.set_params(**{k: v})
        else:
            for val in v:
                clf = classifier.set_params(**{k: val})
    print(clf)
    pprint(clf.get_params())
    clf.fit(X_train, y_train)
    return clf


def classify(X_train, y_train, X_test, y_test, params, clf_algo, clf_name, method):
    '''Dispatcher:
    1. send to train_classifier to get model
    2. send to compute & plot confusion matrix
    3. optionally send to plot ROC & AUC curves
    '''
    handles = get_handles()
    clf     = train_classifier(clf_algo, params, X_train, y_train)
    result_dispatcher(clf, X_test, y_test, np.array(handles), clf_name, method)
    if ROC_flag: ROC(clf, clf_name, method, X_train, X_test, y_train, y_test, save=True)
    return clf


### HYPER-PARAMETER TUNING PARAMETER-SETTING FUNCTIONS ###
def set_common_parameters():
    '''Sets & returns base parameters common to *all* classifiers'''
    parameters = {'random_state': [RANDOM_STATE],
                  'verbose'     : [VERBOSE],}
    return parameters


def load_opt_parameters(key, method, randomized):
    '''Reads in & returns optimal parameters from previously computed/stored grid search
    stored as file
        randomized = True : use results from RandomizedSearchCV()
        randomized = False : use results from GridSearchCV()
    '''
    model_type = 'tuning'
    if randomized: model_type  += '_random' # load best model from RandomizedSearchCV
    exh_model = load_model(key, method, model_type)
    return exh_model.best_params_


def set_opt_parameters(key):
    '''Same as load_opt_parameters() but parameters entered manually rather than from file'''
    if key == 'LR':
        params = set_common_parameters()
        params.update({'C'           : [0.125],
                       'class_weight': [None],
                       'l1_ratio'    : [0.13333333333333333],
                       'max_iter'    : [2500],
                       'penalty'     : ['elasticnet'],
                       'solver'      : ['saga'],
                       'tol'         : [0.001],})
    return params


def build_RandomizedSearch_grid(key):
    '''Builds default parameters for random search'''
    parameter_grid = list()
    params         = set_common_parameters()
    if key == 'LR':
        params.update({'C'            : np.logspace(-7.0, 0., num=10),
                       'solver'       : ['sag', 'saga', 'lbfgs', 'newton-cg', 'liblinear'],
                       'penalty'      : ['l1', 'l2', 'elasticnet', None],
                       'l1_ratio'     : np.linspace(0.05, 0.95, num=10),
                       'class_weight' : ['balanced', None],
                       'max_iter'     : [2500],
                       'tol'          : [1.0e-3],})
    elif key == 'SVM':
        params.update({'C'            : np.logspace(-3.0, 3.0, num=100),
                       'kernel'       : ['rbf', 'poly', 'sigmoid', 'linear'],
                       'gamma'        : ['scale', 'auto'] + [np.logspace(-7, 2, 10)],
                       'class_weight' : ['balanced', None],
                       'max_iter'     : [10000],
                       'tol'          : [1.0e-3],})
    elif key == 'RF':
        params.update({'criterion'        : ['entropy', 'gini'],
                       'bootstrap'        : [True, False],
                       'max_features'     : ['auto', 'log2', None] + [np.linspace(100, 150, 3, dtype = int)],
                       'max_depth'        : [np.linspace(10, 100, 10, dtype = int)] + [None],
                       'min_samples_leaf' : np.linspace(1, 5, 5, dtype = int),
                       'min_samples_split': np.linspace(2, 11, 5, dtype = int),
                       'n_estimators'     : np.linspace(100, 2000, 10, dtype = int),})
    elif key == 'GB':
        params.update({'learning_rate': np.linspace(.4, .6, num=5),
                       'max_depth'    : np.linspace(10, 20, num=10, dtype=int),
                       'n_estimators' : np.linspace(180, 250, num=7, dtype=int),})
    elif key == 'MLP':
        params.update({'solver'            : ['sgd'],
#                       'solver'            : ['adam', 'sgd'],
#                       'activation'        : ['relu', 'tanh', 'logistic'],
                       'activation'        : ['logistic'],
                       'batch_size'        : np.linspace(20, 80, 5, dtype=int),
                       'alpha'             : np.linspace(1, 10.0, 5),
                       'momentum'          : np.linspace(.5, .9, 4),
#                       'hidden_layer_sizes': np.linspace(3, 200, 4, dtype=int),
                       'hidden_layer_sizes': [(100), (100, 100), (50, 100, 50)],
                       'learning_rate_init': np.linspace(1e-3, 1e-2, 3),})
    else:
        raise ValueError(f'[build_RandomizedSearch_grid] Unsupported key: {key}')
    pprint(params)
    parameter_grid.append(params)
    return parameter_grid


def build_GridSearch_grid(key, method, model_type):
    '''Builds parameters for exhaustive grid search GridSearchCV()
    The parameters are built to fine-tune results around optimal parameters from RandomizedSearchCV()
    '''
    #read in best model from RandomizedSearchCV
    model  = load_model(key, method, model_type + '_random')
    params = model.best_params_
    if key == 'LR':
        C      = model.best_params_['C']
        params.update({'C' : np.linspace(0.00001, 0.00005, num=10),})
    elif key == 'SVM':
        C       = model.best_params_['C']
        params  = listify_dict_values(params)
        params.update({'C': np.linspace(C - 5, C + 5, num = 25),})
    elif key == 'RF':
        max_z     = model.best_params_['max_depth']
        n_est     = model.best_params_['n_estimators']
        min_split = model.best_params_['min_samples_split']
        print(max_z, n_est, min_split)
        params    = listify_dict_values(params)
        if max_z != None:
            params.update({'max_depth'    : np.linspace(max_z - 10, max_z + 10, num = 5, dtype = int)})
        params.update({'n_estimators'     : np.linspace(n_est - 75, n_est + 75, num = 10, dtype = int)})
        params.update({'min_samples_split': np.linspace(min_split - 2, min_split + 2, num = 5, dtype = int)})
        #params.update({'min_samples_leaf' : np.linspace(1, 3, num=3, dtype = int)})
    return params


if __name__ == '__main__':
    t0 = time.time()
    print(f'\n[tweet_classifier] start time: {time.asctime( time.localtime(t0))}')
    # Method can be either TFIDF or GOW (graph of words)
    stemmer_flag = False # True: perform stemming
    reset        = False # True to load+preprocess data / False to load processed data from file
    save         = True  # save models to disk
    RANDOM_STATE = 42
    VERBOSE      = 1
    test_size    = .2    # train/test splitting ratio
    ROC_flag     = False # compute ROC/AUC & make plot

    # Controls:
    # Logistic regression / SVM / Random forest / Boosted trees / Neural network
    clf_flags  = {'LR': True, 'SVM': False, 'RF': False, 'GB': False, 'MLP': False}
    model_type = 'base' # can be 'base'(default params), 'tuning' (randomized or grid search) or 'opt' (optimal)
    randomized = True     # grid search (False) or randomized search (True)
    method     = 'TFIDF'    # TFIDF or GOW
    manual_opt = False    # Read opt parameters from file (False) or set them manually (True)
    describe_run(clf_flags, method, model_type, randomized, 'start')

    # Get list of algorithms and their names
    clf_algos = get_classifiers()
    clf_names = get_classifier_names()

    # Load the data from the 3 csv files or read saved processed data
    if reset == False : # Data already processed and saved / reset flag False
        print('*** [main] Loading saved processed data from file ***')
        corpus, labels = load_processed_data()
    else: # Load and process raw data
        raw_data, raw_labels = load_data_from_file()
        print(f'[main] Raw data dims - tweets: {len(raw_data)} / labels: {len(raw_labels)}')

        # Pre-process tweets & labels:
        corpus, labels = preprocess(raw_data, raw_labels, stemmer_flag)

        store_processed_data(corpus, labels) # save processed data to file
    print(f'[main] Processed data dims - corpus: {len(corpus)} / labels: {len(labels)}')

    # Compute TF-IDF or Graph of Words
    if method == 'TFIDF':
        X = tf_idf(corpus)
    elif method == 'GOW':
        X = gow(corpus)
    else:
        raise ValueError(f'[main] Invalid method: {method}')

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        labels,
                                                        test_size    = test_size,
                                                        stratify     = labels,
                                                        random_state = RANDOM_STATE,)
    print(f'[main] Training/test set dimensions: {len(y_train)}/{len(y_test)}\n')

    # Normalize Xs:
    X_train, X_test = normalize_X(X_train, X_test)

    for key, flag in clf_flags.items(): # Main loop: over selected algorithms
        if flag: run_algorithm(key, method, model_type, randomized, X_train, y_train, X_test, y_test)

    describe_run(clf_flags, method, model_type, randomized, 'end')
    print(f'\n[tweet_classifier] {time.asctime( time.localtime(time.time()) )} \
          running time {(time.time() - t0):.1f}s')