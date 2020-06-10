#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 13:51:25 2020

gow_analysis.py

Graph of words analysis:
K-core
K-truss
Max clique
Graph metrics

@author: charles mégnin
"""
print(__doc__)

import time
import networkx as nx

from tweet_io import load_data_by_name
from tweet_processor import preprocess
from handles import get_name_dict, get_color_dict, get_handles
from nx_helper import graph_characteristics, max_ktruss, get_difference, extract_max_clique, graph_commonality, create_corpus_graph_of_words
from gow_plots import plot_commonality, plot_unicity, plot_gow, plot_max_clique
from plot_helper import build_title

plot_directory = './plots/gow/'

def commonality_driver(Gx, name_dict, handles, dense_flag, window, color):
    '''
    Compute nodes common to graphs
    I[0] = I_01 / I[1] = I_02 / I[2] = I_12
    1. Compute nodes common to each possible pair
    2. Compute nodes common to all handles
    3. Compute nodes exclusively common to each possible pair (ie: 1. - 2.)
    '''
    layout_ = 'spring'
    # 1. nodes common to all pairs
    Ints = list() # list of nodes common to all possible pairs

    for i in range(3): # incusive common
        for j in range(i, 3):
            if i != j:
                print(f'commonality_driver: {handles[i]}-{handles[j]}')
                I_temp = graph_commonality(Gx[i], Gx[j])
                Ints.append(I_temp)
                # do not save this plot
                plot_commonality(I_temp,
                                 name_dict,
                                 [handles[i], handles[j]],
                                 window,
                                 color,
                                 dense_flag,
                                 'shell',
                                 False)

    # 2. nodes common to all handles
    print('commonality_driver: Computing commonality between all handles')
    I_012 = graph_commonality(Ints[0], Gx[2])
    print(I_012.nodes)
    plot_commonality(I_012,
                     name_dict,
                     [handles[0], handles[1], handles[2]],
                     window,
                     commonality_color,
                     dense_flag,
                     layout_,
                     True)

    # 3. nodes exclusively common to all pairs
    I_x = list()
    for i in range(3): # exclusive common
        # nodes common to a pair excluding those common to all 3
        I_temp = get_difference(Ints[i], I_012)
        I_x.append(I_temp)
        if   i == 0 : names = [handles[0], handles[1]]
        elif i == 1 : names = [handles[0], handles[2]]
        elif i == 2 : names = [handles[1], handles[2]]
        else: raise ValueError('Invalid iteration:', i)

        plot_commonality(I_temp,
                         name_dict,
                         names,
                         window,
                         x_color,
                         dense_flag,
                         layout_,
                         True)

    return Ints, I_x


def unicity_driver(Gx, Ints, name_dict, names, dense_flag, window):
    layout_ = 'spring'
    Unique  = list()

    color = color_dict[names[0]]
    Sp1 = get_difference(Gx[0], Ints[0])
    Sp2 = get_difference(Sp1, Ints[1])
    Unique.append(Sp2)
    plot_unicity(Sp2, name_dict, names[0], dense_flag, window, layout_, color)

    color = color_dict[names[1]]
    Sp1 = get_difference(Gx[1], Ints[0])
    Sp2 = get_difference(Sp1, Ints[2])
    Unique.append(Sp2)
    plot_unicity(Sp2, name_dict, names[1], dense_flag, window, layout_, color)

    color = color_dict[names[2]]
    Sp1 = get_difference(Gx[2], Ints[1])
    Sp2 = get_difference(Sp1, Ints[2])
    Unique.append(Sp2)
    plot_unicity(Sp2, name_dict, names[2], dense_flag, window, layout_, color)

    return Unique


def run_Gdense(Gc, name, flag):
    if flag == 'kcore':
        G_dense = nx.k_core(Gc)
        prefix = 'Main core'
    elif flag == 'ktruss':
        itruss, G_dense = max_ktruss(Gc, min_truss, max_truss)
        prefix = str(itruss) + '-truss'

    n_nodes = nx.number_of_nodes(G_dense)
    n_edges = nx.number_of_edges(G_dense)
    print(f'k-core: {n_nodes} nodes / {n_edges} edges')

    suffix = str(n_nodes) + ' nodes'
    # suffix += ' window=' + str(window_size)

    title = build_title(prefix, name, suffix)
    plot_gow(G_dense, title, color, layout_, True)

    return G_dense


if __name__ == '__main__':
    t0 = time.time()
    stemmer_flag      = False # stemming
    commonality_color = 'olivedrab' # color for common to all
    x_color           = 'gold' # color for unique nodes
    window_size       = 5
    min_truss         = 2   # start at 2 if not too CPU-intensive
    max_truss         = 100 # can be set arbitrarily large

    # Flags
    max_clique_flag        = True # plot max clique
    max_clique_corpus_flag = False # for corpus: max_clique is CPU-intensive

    centre_flag      = False # compute distance measures: center, radius, diameter
    corpus_dist_flag = False # distance characteristics (radius, center) are CPU-intensive

    kcore_flag   = False # compute main core
    ktruss_flag  = True # compute k-truss

    handles    = get_handles()
    name_dict  = get_name_dict()
    color_dict = get_color_dict()
    print(f'--- kcore: {kcore_flag} ktruss: {ktruss_flag} window={window_size} ---')

    Gs_corpus = list() # corpus graphs
    Gs_kcore  = list() # k-core graphs
    Gs_ktruss = list() # k-truss graphs
    for handle in handles: # Loop over handles
        layout_ = 'spring'
        color   = color_dict[handle] # define color for the current handle
        name    = name_dict[handle]  # define current handle readable name
        print(f'\nPre-processing {name}...')

        # Load tweets from file
        raw_tweets, class_labels = load_data_by_name(handle)
        tweets, class_labels = preprocess(raw_tweets, class_labels, stemmer_flag)

        # Extract graphs from corpus of tweets
        G_corpus = create_corpus_graph_of_words(tweets, window_size, False)
        Gs_corpus.append(G_corpus)
        if max_clique_corpus_flag:
            corpus_time = time.time()
            print(f'[main]: Computing max clique for whole corpus')
            Gmax_clique = extract_max_clique(G_corpus)
            print(f'corpus max clique computing time - {(time.time() - corpus_time):.1f}s')
            print(f'plotting...', end=' ')
            plot_max_clique(Gmax_clique, name_dict, handle, 'corpus', color, layout_)
            print(f'corpus max clique total time - {(time.time() - corpus_time):.1f}s')

            print(f"\nCorpus characteristics: {name}:")
            graph_characteristics(G_corpus, corpus_dist_flag)

        if kcore_flag: # Extract main k-core
            G_kcore = run_Gdense(G_corpus, name, 'kcore')
            Gs_kcore.append(G_kcore)
            # Compute max clique
            if max_clique_flag:
                print(f'[main]: Computing max clique for kcore')
                Gmax_clique = extract_max_clique(G_kcore)
                plot_max_clique(Gmax_clique, name_dict, handle, 'kcore', color, layout_)
            print(f"\nK-core characteristics: {name}")
            # Compute graph characteristics
            graph_characteristics(G_kcore, True)

        if ktruss_flag: # Extract maximum k-truss
            G_truss = run_Gdense(G_corpus, name, 'ktruss')
            Gs_ktruss.append(G_truss)
            # Compute max clique
            if max_clique_flag:
                print(f'[main]: Computing max clique for ktruss')
                Gmax_clique = extract_max_clique(G_truss)
                plot_max_clique(Gmax_clique, name_dict, handle, 'ktruss', color, layout_)
            print(f"\nK-truss characteristics: {name}")
            # Compute graph characteristics
            graph_characteristics(G_truss, True)
            # *** END LOOP OVER HANDLES ***

    if kcore_flag: # Compute and plot common and unique nodes
        dense_flag = 'kcore'
        print(f'[main]: Computing common {dense_flag} nodes {handle}')
        Ints, I_x = commonality_driver(Gs_kcore,
                                       name_dict,
                                       handles,
                                       dense_flag,
                                       window_size,
                                       commonality_color)
        print(f'[main]: Computing unique {dense_flag} nodes {handle}')
        Unique = unicity_driver(Gs_kcore, Ints, name_dict, handles, dense_flag, window_size)

    if ktruss_flag: # Compute and plot common and unique nodes
        dense_flag = 'ktruss'
        print(f'[main]: Computing common {dense_flag} nodes {handle}')
        Ints, I_x = commonality_driver(Gs_ktruss,
                                       name_dict,
                                       handles,
                                       dense_flag,
                                       window_size,
                                       commonality_color)
        print(f'[main]: Computing unique {dense_flag} nodes {handle}')
        Unique = unicity_driver(Gs_ktruss, Ints, name_dict, handles, dense_flag, window_size)

    print(f'\ngow_analysis: {time.asctime( time.localtime(time.time()) )}  running time --- {(time.time() - t0):.1f} seconds ---\n')