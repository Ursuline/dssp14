#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:57:08 2020

tweet_eda.py

@author: charles mégnin
"""
print(__doc__)

import time
import numpy as np
from scipy import stats

from handles import get_handles, get_name_dict, get_color_dict, get_names
from tweet_io import load_data_by_name
from tweet_processor import preprocess

import eda_plots

def dic_to_array(dic):
    '''
    returns numpy array from nwords: count dictionary dic
    counts are unrolled, ie: 4:3 -> 4, 4, 4
    '''
    l = list()
    for nwords, counts in dic.items():
        for i in range(counts):
            l.append(nwords)
    return np.array( l )


def word_count_ttest(arr, names):
    '''t-test for difference in number of words'''
    for i in range(3):
        for j in range(i+1, 3):
            t2, p2 = stats.ttest_ind(arr[i],arr[j])
            print(f"t {names[i]}-{names[j]}= {str(t2)}")
            print(f"p {names[i]}-{names[j]}= {str(p2)}\n")


def build_word_dict(tweets):
    '''builds dictionary of {nwords/tweet : count}'''
    wc_tweet = dict()
    for tweet in tweets:
        nwords = len(tweet.split())

        #build histogram :
        if nwords in wc_tweet:
            wc_tweet[nwords] += 1
        else:
            wc_tweet[nwords] = 1

    return wc_tweet

def build_char_dict(tweets):
    '''builds dictionary of {nchars/word : count}'''
    cc_tweet = dict()
    for tweet in tweets:
        for word in tweet.split():
            nchar = len(word)

            #build histogram :
            if nchar in cc_tweet:
                cc_tweet[nchar] += 1
            else:
                cc_tweet[nchar] = 1

    return cc_tweet


def build_set(tweets):
    '''builds set of (unique) words from corpus of tweets'''
    unique_words = set()
    for tweet in tweets:
        for word in tweet.split():
            unique_words.add(word)

    return unique_words


if __name__ == '__main__':
    t0 = time.time()
    name_dict  = get_name_dict()
    color_dict = get_color_dict()
    handles    = get_handles()
    names      = get_names()

    start_dates = ['05/01/2018', '03/07/2018', '02/02/2019']
    end_dates   = ['25/01/2020', '26/01/2020', '25/01/2020']
    eda_plots.tweeting_period_plot(handles,
                                   start_dates,
                                   end_dates,
                                   color_dict)

    nraw    = list() # number of raw tweets for each handle
    ntweets = list() # number of processed tweets for each handle
    word_counts  = list() # list of word counts
    char_counts  = list() # list of character counts
    unique_words = list() # list of # of unique words for each handle
    unique_all   = set()  # set of unique words overall
    wc_arrays    = list() # list of arrays of word_counts for sig. test

    for handle in handles: # Loop over handles
        print(name_dict[handle])
        raw_tweets, class_labels = load_data_by_name(handle)
        tweets, labels = preprocess(raw_tweets, class_labels, False)
        nraw.append(len(raw_tweets))
        ntweets.append(len(tweets))

        print(f'raw tweets: {len(raw_tweets)}', end=' ')
        print(f'processed tweets: {len(tweets)}')

        # words/tweet counts : tweet length
        wc_tweet = build_word_dict(tweets)
        word_counts.append(wc_tweet)
        ar = dic_to_array(wc_tweet) # numpy array of word counts
        print(stats.describe(ar))

        wc_arrays.append(ar)

        eda_plots.plot_distribution(wc_tweet,
                                    name_dict[handle],
                                    'Tweet length',
                                    '# words / tweet',
                                    35,
                                    300,
                                    color_dict[handle],
                                    np.mean(ar),
                                    np.std(ar))

        # unique words
        unique_set = build_set(tweets)
        unique_words.append(len(unique_set))
        print(f'{len(unique_set)} unique words\n')
        unique_all.update(unique_set)

        #character/word counts
        cc_tweet = build_char_dict(tweets)
        char_counts.append(cc_tweet)
        ar = dic_to_array(cc_tweet) # numpy array of word counts
        print(cc_tweet)
        eda_plots.plot_distribution(cc_tweet,
                                    name_dict[handle],
                                    'Word length',
                                    '# characters / word',
                                    24,
                                    7000,
                                    color_dict[handle],
                                    np.mean(ar),
                                    np.std(ar))

    print(f'Unique overall = {len(unique_all)}')

    eda_plots.word_volume_plot(handles,
                               word_counts,
                               color_dict)

    eda_plots.unique_word_volume_plot(handles,
                                      unique_words,
                                      color_dict)

    # diversity = unique words / total words
    diversity = list()
    for i in range(len(handles)):
        temp = 0
        for key, value in word_counts[i].items():
            temp += int(key*value)
        diversity.append(unique_words[i]/temp)

    eda_plots.diversity_plot(handles,
                             diversity,
                             color_dict)

    # unique words / tweet
    uwpt = [ (unique_words[i]/ntweets[i]) for i in range(len(handles)) ]

    eda_plots.uwpt_plot(handles,
                        uwpt,
                        color_dict)

    # Tweet frequency
    eda_plots.tweet_frequency_plot(handles,
                                   start_dates,
                                   end_dates,
                                   ntweets,
                                   color_dict)

    eda_plots.tweet_count_plot(nraw,
                               ntweets,
                               names)

    print(f'\ntweets_eda: {time.asctime( time.localtime(time.time()) )} Running time: {(time.time() - t0):.1f} seconds\n')