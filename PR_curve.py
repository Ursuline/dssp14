#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 21:26:52 2020

PR_curve.py

Precision - recall curve

@author: charles mégnin
"""
print(__doc__)
import matplotlib.pyplot as plt
import numpy as np
import string
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle

from classifier_helper import get_classifier, normalize_X, get_classifier_name
from tweet_io import load_processed_data, clean_filename, load_model
from graph_of_words import gow
from tfidf import tf_idf
from handles import get_color_dict, get_handles

plot_dir = 'plots/prediction/'

def plot_PR_curve():
    #Plot the micro-averaged Precision-Recall curve
    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-recall {algo} {method} average score: {average_precision["micro"]:0.2f}')
    plt.show()


def plot_PR_curves(save=True):
    #Plot Precision-Recall curve for each class and iso-f1 curves
    # setup plot details
    legend_size = 12
    title_size  = 18
    color_dict  = get_color_dict()
    colors      = cycle([list(color_dict.values())[0],
                         list(color_dict.values())[1],
                         list(color_dict.values())[2],
                         'darkorange',
                         'teal'])

    clf = get_classifier_name(algo)
    title = f'{method}+{clf}'

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines  = []
    labels = []

    # plot iso-f1 curves
    for f_score in f_scores:
        x  = np.linspace(0.01, 1)
        y  = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate(f'f1={f_score:0.1f}', xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append(f'micro-average precision-recall (area = {average_precision["micro"]:0.2f})')

    for i, color in zip(range(n_classes), colors):
        handle = string.ascii_uppercase[i]
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append(f'Precision-recall for handle {handle} (area = {average_precision[i]:0.2f})')

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title, fontsize=title_size)
    plt.legend(lines, labels, loc=(0, -.37), prop=dict(size=legend_size))
    if save:
        title = f'PR {algo} {method}'
        plt.savefig(clean_filename(title , 'png', plot_dir))
    plt.show()


if __name__ == '__main__':
    RANDOM_STATE   = 42
    n_jobs         = 8
    algo           = 'MLP'
    method         = 'TFIDF'
    test_size      = .2

    estimator = get_classifier(algo)
    X, y      = load_processed_data()

    # create multi-label like settings
    handles = get_handles()
    y = label_binarize(y, classes= [handles[0], handles[1], handles[2]])

    n_classes = y.shape[1]

    # Compute TF-IDF or Graph of Words
    if method == 'TFIDF' :
        X = tf_idf(X)
    elif method == 'GOW':
        X = gow(X)
    else:
        raise ValueError(f'[main] Invalid method: {method}')

    # Split into training and test
    X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                        y,
                                                        test_size    = test_size,
                                                        random_state = RANDOM_STATE)
    X_train, X_test = normalize_X(X_train, X_test)

    # Load the best model from GridSearchCV()
    model = load_model(algo, method, 'tuning_random')
    params = model.best_params_
    print(f'Best model parameters: \n{params}\n')
    #params.update({'n_jobs': n_jobs},)

    # OneVsRestClassifier for multi-label prediction
    # Run classifier
    # ***Change the classifier here:***
    classifier = OneVsRestClassifier(MLPClassifier(**params))
    classifier.fit(X_train, Y_train)
    if algo in ('LR', 'SVM'):
        y_score = classifier.decision_function(X_test)
    else:
        y_score = classifier.predict_proba(X_test)


    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
                                                                    y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                         average="micro")
    print(f'Average precision score micro-averaged over all classes: {average_precision["micro"]:0.2f}')

    plot_PR_curve()
    plot_PR_curves()