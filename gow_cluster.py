#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:46:03 2020

gow_cluster.py extracted from bloated gow_analysis.py

Spectral clustering of either corpus, main core or main k-truss

@author: charles mégnin
"""
import time

from tweet_processor import preprocess
from tweet_io import load_data_by_name
from handles import get_name_dict, get_color_dict, get_handles
from nx_helper import create_corpus_graph_of_words
from clustering_helper import process_spectral_clustering
from nx_helper import extract_Gdense

def run_info():
    print(f'corpus={corpus_flag} kcore={kcore_flag} ktruss={ktruss_flag}')
    print(f'window={window} / {n_clusters} clusters\n')


if __name__ == '__main__':
    t0 = time.time()
    stemmer_flag = False # stemming
    window      = 5

    # Controls:
    corpus_flag = True # compute main core
    kcore_flag  = False # compute main core
    ktruss_flag = False # compute k-truss
    n_clusters  = 3 # number of spectral clusters

    run_info() # run info to stdout

    handles     = get_handles()
    name_dict   = get_name_dict()
    color_dict  = get_color_dict()

    Gs_kcore  = list() # k-core graphs
    Gs_ktruss = list() # k-truss graphs
    corpus    = list()
    labels    = list()
    for handle in handles: # Loop over handles
        # Load tweets from file
        raw_tweets, class_labels = load_data_by_name(handle)
        tweets, class_labels = preprocess(raw_tweets, class_labels, stemmer_flag)
        corpus += tweets
        labels += class_labels
    print(f'corpus size: {len(corpus)} labels size = {len(labels)}')

    # Extract graph from corpus of tweets
    G_corpus = create_corpus_graph_of_words(corpus, window, False)
    if corpus_flag: # Extract main k-core
        process_spectral_clustering(G_corpus,
                                    n_clusters,
                                    'all handles',
                                    'olivedrab',
                                    'corpus',)

    if kcore_flag: # Extract main k-core
        G_kcore = extract_Gdense(G_corpus, 'kcore')
        Gs_kcore.append(G_kcore)
        process_spectral_clustering(G_kcore,
                                    n_clusters,
                                    'all handles',
                                    'darkorange',
                                    'kcore',)

    if ktruss_flag: # Extract maximum k-truss
        G_truss = extract_Gdense(G_corpus, 'ktruss')
        Gs_ktruss.append(G_truss)
        process_spectral_clustering(G_truss,
                                    n_clusters,
                                    'all handles',
                                    'aqua',
                                    'ktruss',)

    print(f'\ngow_cluster: {time.asctime( time.localtime(time.time()) )}  running time --- {(time.time() - t0):.1f} seconds ---\n')
