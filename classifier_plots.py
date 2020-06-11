#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 04:27:02 2020

classifier_plots.py

@author: charles mégnin
"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import confusion_matrix
from handles import get_color_dict, get_names, get_handles
from tweet_io import clean_filename

plot_directory = './plots/prediction/'

def plot_confusion_matrix(y_pred, y_test, score, handles, algo, method, normalize, save=True):
    """
    Plot a confusion matrix
    Expected True values x-axis & predicted y-axis
    obtained from : confusion_matrix(y_pred, y_test, labels = labels)
    note: this order is the opposite of that suggested in scikit-learn
    """
    figsize_x = 6.5
    figsize_y = 6
    cm = confusion_matrix(y_pred, y_test, labels = handles)
    print(cm)
    print(f'[confusion_matrix_wrapper] {algo} score = {score}\n')
    labels = get_names()
    #print(f'[plot_confusion_matrix] labels = {labels}')
    accuracy = np.trace(cm) / float(np.sum(cm))
    print(f'[plot_confusion_matrix] Accuracy={accuracy}')
    print(f'[plot_confusion_matrix] confusion matrix:\n{cm}')
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm, 2)

    fig = plt.figure(figsize=(figsize_x, figsize_y))
    ax = fig.add_subplot(111)
    ax.matshow(cm)

    #if method == 'GOW': method = 'Graph of Words'
    title = f'{method}+{algo}'
    plt.title(title, y=1.1)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.ylabel('Predicted')
    plt.xlabel(f'True\naccuracy={accuracy:0.3f}')
    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(cm[i][j]))
    if save:
        plt.savefig(clean_filename(title , 'png', plot_directory))
    plt.show()


def ROC_plot(fpr, tpr, roc_auc, n_classes, ix, gmean, algo, method, save=True):
    '''Plots ROC curves
    gmean = average
    '''
    color_dict = get_color_dict()
    names      = get_names()
    handles    = get_handles()

    lw      = 1
    figsize = 6
    title   = f'ROC {method}+{algo}'

    # Make plot
    plt.figure(figsize=(figsize, figsize))
    colors = cycle([color_dict[handles[0]],
                    color_dict[handles[1]],
                    color_dict[handles[2]]])

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, linewidth=lw,
                 label=f'ROC {names[i]} (auc = {roc_auc[i]:0.2f})')
        if i == 0:
            plt.scatter(fpr[i][ix[i]], tpr[i][ix[i]], marker='o', color='black', label=f'Best (gmeans={gmean:.3f})')
        else:
            plt.scatter(fpr[i][ix[i]], tpr[i][ix[i]], marker='o', color='black')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=lw, label='No Skill') # diagonal

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    if save:
        plt.savefig(clean_filename(title , 'png', plot_directory))
    plt.show()
