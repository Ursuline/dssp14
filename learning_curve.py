#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 18:15:55 2020

@author: charly
"""

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
#from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split

from tweet_io import load_processed_data
from classifier_helper import normalize_X

from graph_of_words import gow
from tfidf import tf_idf

plot_dir = 'plots/prediction/'

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    samples_fit_time_flag = False
    fit_score_flag = False

    #if axes is None:
        #_, axes = plt.subplots(1, 3, figsize=(20, 5))
       # _, axes = plt.subplots(1, 1, figsize=(12, 13))
    fig = plt.figure(figsize=(12, 13))

    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - 2*train_scores_std,
                         train_scores_mean + 2*train_scores_std,
                         alpha=0.1,
                         color="r")
    plt.fill_between(train_sizes, test_scores_mean - 2*test_scores_std,
                         test_scores_mean + 2*test_scores_std,
                         alpha=0.1,
                         color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="CV score")
    plt.legend(loc="best")

    # Plot n_samples vs fit_times
    if samples_fit_time_flag:
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-')
        axes[1].fill_between(train_sizes, fit_times_mean - 2*fit_times_std,
                             fit_times_mean + 2*fit_times_std, alpha=0.1)
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    if fit_score_flag :
        axes[2].grid()
        axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
        axes[2].fill_between(fit_times_mean, test_scores_mean - 2*test_scores_std,
                             test_scores_mean + 2*test_scores_std, alpha=0.1)
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")

    filename = plot_dir + f'learning_curve_{algo_name}_{method}.png'
    plt.savefig(filename)

    return plt

if __name__ == '__main__':
    RANDOM_STATE = 42
    n_jobs       = 2
    y_min        = .6
    algo_name    = 'LR' # LR or SVC
    method       = 'TFIDF'
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))

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

    if algo_name == 'LR':
        title = f"Learning Curves LR+{method}"
        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        #cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=RANDOM_STATE)
        cv = 10
        if method   == 'TFIDF': C = 1.29e-4
        elif method == 'GOW'  : C = 2.15e-5
        estimator = LogisticRegression(solver = 'sag', penalty = 'l2', C=C)
        plot_learning_curve(estimator, title, X, y, ylim=(y_min, 1.01),
                            cv=cv, n_jobs=n_jobs)

    elif algo_name == 'SVC':
        title = rf"Learning Curves SVC+{method}, sigmoid kernel"
        # SVC is more expensive so we do a lower number of CV iterations:
        #cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=RANDOM_STATE)
        cv = 10
        if method   == 'TFIDF': C = 2.477
        elif method == 'GOW'  : C = 1.630
        estimator = SVC(gamma  = 'scale', kernel = 'sigmoid', C=C)
        plot_learning_curve(estimator, title, X, y, ylim=(y_min, 1.01),
                            cv=cv, n_jobs=n_jobs)

    plt.show()