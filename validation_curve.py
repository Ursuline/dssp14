#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:00:41 2020

validation_curve.py

@author: charles mégnin
"""
print(__doc__)

import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
from sklearn.model_selection import train_test_split

from tweet_io import load_processed_data
from classifier_helper import normalize_X
from graph_of_words import gow
from tfidf import tf_idf

plot_dir = 'plots/prediction/'

scoring      = 'accuracy'
RANDOM_STATE = 42

param_name   = 'C'
method       = 'GOW'
algo_name    = 'SVC'

X, y       = load_processed_data()
# Compute TF-IDF or Graph of Words
if method == 'TFIDF' :
    X = tf_idf(X)
elif method == 'GOW':
    X = gow(X)
else:
    raise ValueError(f'[main] Invalid method: {method}')

# Split the data
X, X_test, y, _ = train_test_split(X,
                                   y,
                                   test_size    = .2,
                                   stratify     = y,
                                   random_state = RANDOM_STATE,)
X, _ = normalize_X(X, X_test)

if algo_name == 'SVC':
    param_range = np.logspace(-2, 3, 6)
    train_scores, test_scores = validation_curve(SVC(gamma  = 'scale',
                                                     kernel = 'sigmoid'),
                                                 X,
                                                 y,
                                                 param_name  = param_name,
                                                 param_range = param_range,
                                                 scoring     = scoring,
                                                 n_jobs      = 2)
elif algo_name == 'LR':
    param_range = np.logspace(-7, 1, 9)
    train_scores, test_scores = validation_curve(LogisticRegression(solver  = 'sag',
                                                                    penalty = 'l2'),
                                                 X,
                                                 y,
                                                 param_name  = param_name,
                                                 param_range = param_range,
                                                 scoring     = scoring,
                                                 n_jobs      = 2)
else: raise ValueError(f'[main] {algo_name} not implemented')


train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std  = np.std(train_scores,  axis=1)
test_scores_mean  = np.mean(test_scores,  axis=1)
test_scores_std   = np.std(test_scores,   axis=1)

plt.title(f"Validation Curve {method} / {algo_name}")
plt.xlabel(rf"${param_name}$")
plt.ylabel(f"{scoring}")
plt.ylim(0.0, 1.1)
lw    = 0.5
alpha = 0.2

plt.semilogx(param_range,
             train_scores_mean,
             label = "Training score",
             color = "darkorange",
             lw    = lw)

plt.fill_between(param_range,
                 train_scores_mean - 2.0 * train_scores_std,
                 train_scores_mean + 2.0 * train_scores_std,
                 alpha = alpha,
                 color = "darkorange",
                 lw    = lw)

plt.semilogx(param_range,
             test_scores_mean,
             label = "CV score",
             color = "navy",
             lw    = lw)

plt.fill_between(param_range,
                 test_scores_mean - 2.0 * test_scores_std,
                 test_scores_mean + 2.0 * test_scores_std,
                 alpha = alpha,
                 color = "navy",
                 lw    = lw)

plt.legend(loc = "best")

filename = plot_dir + f'validation_curve_{method}_{algo_name}.png'
plt.savefig(filename)
plt.show()