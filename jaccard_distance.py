#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 12:44:56 2020

@author: charles mégnin
"""
print(__doc__)

import time
import nltk
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from tweet_io import load_all_tweets, clean_filename
from tweet_processor import iterate_preprocess
from handles import get_handles, get_name_dict

plot_directory = './plots/jaccard/'

def jd_to_heatmap(K, title, max_scale, save=True):

    # Generate a colormap
    cmap = "BuGn"
    x = [0.5, 1.5, 2.5]

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(K,
                cmap=cmap,
                vmax=max_scale,
                vmin= .72,
                #center=0,
                square=True,
                linewidths=.75,
                cbar_kws={'shrink': .75,
                          'label': ''},
                xticklabels=True,
                yticklabels=True,
                annot=True,)
    sns.set(font_scale=2)
    sns.set(style="white")
    plt.xticks(x, names)
    plt.yticks(x, names)
    plt.title(title)
    if save:
        filename = clean_filename(title, 'png', plot_directory)
        print(f'saving plot to {filename}')
        plt.savefig(filename)
    plt.show()


def array_to_matrix(arr):
    mat = np.zeros(shape=(3,3))
    k = 0
    for i in range(3):
        for j in range(i, 3):
            if i == j: mat[i][j] = 1.0
            else:
                mat[i][j] = arr[k]
                mat[j][i] = mat[i][j]
                k += 1
    return mat


if __name__ == '__main__':
    t0 = time.time()
    name_dict = get_name_dict()

    stemmer_flag = False
    max_ngram        = 2

    # Load tweets from csv files
    names = list()
    handles = get_handles()
    for handle in handles:
        names.append(name_dict[handle])

    raw_corpus, raw_labels = load_all_tweets()
    corpus, labels         = iterate_preprocess(raw_corpus,
                                                raw_labels,
                                                handles,
                                                stemmer_flag)

    for ngram in range(1, max_ngram+1): # loop through ngrams
        # Create three sets of tokens
        tokens = list()
        for tweets in corpus:
            tk = list()
            for tweet in tweets:
                tk += nltk.word_tokenize(tweet)
            tokens.append(set(nltk.ngrams(tk, n=ngram)))

        jd = list()
        for i in range(3):
            for j in range(i+1, 3):
                jd.append(nltk.jaccard_distance(tokens[i], tokens[j]))

        mat = array_to_matrix(jd)
        title = 'jaccard distance - ngram=' + str(ngram)
        jd_to_heatmap(mat, title, 1, True)

    print(f'\njaccard distance: Running time: {(time.time() - t0):.1f}s\n')