#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 01:13:57 2020

tweet_io.py

@author: charles mégnin
"""
import csv
import unidecode
import pickle
from handles import get_handles

def save_to_file(file, contents):
    print(f'saving to {file}')
    with open(file, 'w') as tf:
        tf.write(contents)


def load_data_from_file() :
    """
    reads tweets downloaded with twitter_download.py
    and extracts two fields:
    data are the tweets
    labels are the handles corresponding to each tweet
    returns list of tweets and list of labels
    """
    handles = get_handles()
    directory = './tweets/'
    data   = []
    labels = []

    for handle in handles:
        filename = directory + handle + '_tweets.csv'
        print(f'[load_data_from_file]: Loading {filename}')

        dat, lab = load_data_by_name(handle, directory)

        data   += dat
        labels += lab

        print(f'[load_data_from_file]: cumulative data size= {len(data)} tweets\n')

    return data, labels


def load_data_by_name(handle, directory='./tweets/') :
    """
    reads tweets downloaded with twitter_download.py
    and extracts two fields:
    data are the tweets
    labels are the handles corresponding to each tweet
    """
    filename = directory + handle + '_tweets.csv'

    try:
        nrows = 0
        with open(filename, 'r') as file:
            reader = csv.reader(file, delimiter='|')
            data =[]
            labels = []
            for row in reader:
                # skip missing data
                if row[0] and row[1]:
                    data.append(row[2]) # tweet is 3rd field
                    labels.append(row[0]) # handle is 1st field
                    nrows += 1
    except OSError as e:
        raise e
    return data, labels


def load_all_tweets():
    directory = './tweets/'
    data   = []
    labels = []

    handles = get_handles()

    for handle in handles:
        datum, label = load_data_by_name(handle, directory)
        data.append(datum)
        labels.append(label)
    print(f'[load_all_tweets] tweets loaded\n')
    return data, labels


def clean_filename(name, extension, directory):
    '''
    Makes filenames processable:
    1. replace blank spaces with underscores
    2. remove accents
    3. add directory
    '''
    filename = name.replace(' ', '_') # replace spaces with underescores
    filename = unidecode.unidecode(filename) # remove accents
    return directory + filename + '.' + extension


# Store and load classifiers & processed data to directory model_dir
model_dir = './temp/'

def store_model(clf, algo, method, model_type, randomized):
    '''Stores results from gridsearchcv randomizedsearchcv as pickle file
       * model_type = base, tuning or opt
       * method = TFIDF or GOW
       * algo = LR, RF, etc
       * clf is classifier
       * randomized -> False: model from GridSearchCV / True: model from RandomizedSearchCV
    '''
    mdl = model_type
    if randomized == True: mdl += '_random'
    pkl_filename = f"{algo}_{method}_{mdl}_model.pkl"
    with open(model_dir+pkl_filename, 'wb') as file:
        pickle.dump(clf, file)
    print(f'*** [store_model] model saved as {pkl_filename} ***')


def load_model(algo, method, model_type):
    '''Loads results from GridSearchCV or RandomizedSearchCV stored as pickle file
       (see store_model)
    '''
    pkl_filename = f"{algo}_{method}_{model_type}_model.pkl"
    print(f'[load_model] loading {pkl_filename}')
    try:
        with open(model_dir+pkl_filename, 'rb') as file:
            pickle_model = pickle.load(file)
    except OSError as e:
        raise e
    return pickle_model


def store_processed_data(corpus, labels):
    '''Stores data pre-processed with tweet_core.py > preprocess() as pickle file'''
    pkl_filename = 'processed_corpus.pkl'
    with open(model_dir + pkl_filename, 'wb') as file:
        pickle.dump(corpus, file)
    pkl_filename = 'processed_labels.pkl'
    with open(model_dir + pkl_filename, 'wb') as file:
        pickle.dump(labels, file)


def load_processed_data():
    '''Load pre-processed data saved as a pickle file'''
    pkl_filename = ['processed_corpus.pkl', 'processed_labels.pkl']
    processed_data = []
    try:
        for filename in pkl_filename:
            with open(model_dir+filename, 'rb') as file:
                processed_data.append(pickle.load(file))
    except OSError as e:
        raise e
    return processed_data[0], processed_data[1]


if __name__ == '__main__':
    # raw_tweets, labels = load_all_tweets()
    # print(raw_tweets[0][1],'\n')
    # print(raw_tweets[1][1],'\n')
    # print(raw_tweets[2][1],'\n')
    load_model('junk', 'junk', 'junk')