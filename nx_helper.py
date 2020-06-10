#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:46:48 2020

nx_helper.py

Helper routines for networkx data structures

@author: charles mégnin
"""
import networkx as nx
from networkx.algorithms.approximation import clique

plot_dir = './plots/gow/'

def create_corpus_graph_of_words(tweets, window_size, directed=False):
    # Generates a graph of words for all tweets
    if directed == True:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    for tweet in tweets: # Loop through tweets
        tokens = tweet.split() # tokenize each tweet
        n = len(tokens)
        for i in range(n): #build the edges
            for j in range(i+1, i+window_size):
                if j < n and tokens[i] != tokens[j]:
                    G.add_edge(tokens[i], tokens[j])

    # Add labels as 'label'
    for node in G.nodes():
        G.nodes[node]["label"] = node

    print(f"{G.number_of_nodes()} nodes ", end='')
    print(f"{G.number_of_edges()} edges")

    return G


def tweets_to_nxGraphs(corpus, window, handles):
    print("tweets_to_nxGraphs: Generating graphs from corpus")
    n = len(corpus) # number of handles
    Gs = list()     # list of networkx graphs
    for handle in range(n):
        print(f'tweets_to_nxGraphs: Handle = {handle}')
        G = create_corpus_graph_of_words(corpus[handle], window, False)
        Gs.append(G)

    return Gs


def get_main_cores(G, n):
    print("get_main_cores: Computing main cores")
    G_kcores = list()
    for handle in range(n):
        G_kcore = nx.k_core(G[handle])
        G_kcores.append(G_kcore)

    return G_kcores


def get_ktrusses(G, n):
    print("get_ktrusses: Computing main k-truss")
    min_truss = 2
    max_truss = 100
    G_truss = list()
    for handle in range(n):
        it, G_t = max_ktruss(G[handle], min_truss, max_truss)
        G_truss.append(G_t)

    return G_truss


def graph_commonality(Gx, Hx):
    """
    Computes Nodes and edges that intersect two graphs G & H (= Int)
    """
    return Gx.edge_subgraph(Hx.edges).copy()


def graph_characteristics(G, distance_flag):
    """Output graph characteristics to stdout """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f'{n_nodes} nodes / {n_edges} edges')

    # Compute graph sparsity
    L_max = int( 1/2 * n_nodes* (n_nodes-1) )
    print(f'Lmax = {L_max}')
    print(f"Sparsity: {1.0 - n_edges/L_max:.3f}")

    if distance_flag: # Compute graph radius, diameter and center
        G_center = nx.center(G)
        print(f'Centre size: {len(G_center)}\n {G_center}')
        print(f'Radius: {nx.radius(G)}')
        print(f'Diameter: {nx.diameter(G)}')


def max_ktruss(G, min_truss, max_truss):
    """Computes, plots and returns max k-truss value
        Searches for a maximum k-truss between min_truss and max_truss
        max_truss can be set arbitrarily large
    """
    verbose = 0
    G_save = G
    itruss = min_truss
    # look for max k-truss
    for truss in range(min_truss, max_truss + 1):
        G_truss = nx.k_truss(G, truss)
        if G_truss.number_of_nodes() == 0: # Max k-truss reached
            print(f'Max-truss = {itruss}')
            break
        else:
            G_save = G_truss
            itruss += 1

    if verbose: print(f"Nodes in k-truss : {G_save.nodes()}")

    return itruss, G_save


def get_difference(G, H):
    """Returns nodes in G that are not in H"""
    Diff = G.copy()
    Diff.remove_nodes_from(n for n in G if n in H)

    return Diff


def difference(S, R): # Found online
    DIF = nx.create_empty_copy(R)
    DIF.name = "Difference of (%s and %s)" % (S.name, R.name)
    if set(S) != set(R):
        raise nx.NetworkXError("Node sets of graphs is not equal")

    r_edges = set(R.edges())
    s_edges = set(S.edges())

    # I'm not sure what the goal is: the difference, or the edges that are in R but not in S
    # In case it is the difference:
    diff_edges = r_edges.symmetric_difference(s_edges)

    # In case its the edges that are in R but not in S:
    #diff_edges = r_edges - s_edges

    DIF.add_edges_from(diff_edges)

    return DIF


def extract_max_clique(Gx):
    '''
    Extract the maximum clique from graph Gx
    return max clique as graph
    '''
    max_clique_ = clique.max_clique(Gx)
    G_max = nx.subgraph(Gx, max_clique_)

    return G_max


def cx_subgraph(G_in, node):
    '''Extract subgraph of G_in containing node and its neighbors '''
    nodes = nx.all_neighbors(G_in, node)
    G_out = nx.subgraph(G_in, nodes)
    print(f'cx_subgraph: {G_out.number_of_nodes()} / {G_in.number_of_nodes()} connected to node {node}')
    L_max = G_out.number_of_nodes() * (G_out.number_of_nodes()-1) * 1/2
    n_edges = G_out.number_of_edges()
    print(f'cx_subgraph: Sparcity = {1-n_edges/L_max}')

    return G_out

def extract_Gdense(Gc, flag):
    '''Extract main core of max k-truss from G'''
    if flag == 'kcore':
        G_dense = nx.k_core(Gc)

    elif flag == 'ktruss':
        min_truss, max_truss   = 2, 100 # start at 2 if not too CPU-intensive
        itruss, G_dense = max_ktruss(Gc, min_truss, max_truss)

    n_nodes = nx.number_of_nodes(G_dense)
    n_edges = nx.number_of_edges(G_dense)
    print(f'[extract_Gdense]: {flag} - {n_nodes} nodes / {n_edges} edges')

    return G_dense