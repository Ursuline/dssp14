#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 19:27:20 2020

clustering_helper.py

@author: charly
"""
import numpy as np
import networkx as nx

from random import randint
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigs

from gow_plots import build_title, plot_gow
#from pgv_helper import pgv_plot

plot_directory = './plots/cluster/'

def random_clustering(G):
    random_clustering_result = dict()
    for node in G.nodes():
        random_clustering_result[node] = randint(0, 1)
    return random_clustering_result


def modularity(G, clustering):
    modularity = 0
    m = G.number_of_edges()

    clusters = set(clustering.values())

    for cluster in clusters:
        nodes_in_cluster = [node for node in G.nodes() if clustering[node]==cluster]

        subG = G.subgraph(nodes_in_cluster)
        l_c = subG.number_of_edges()
        d_c = sum([G.degree(node) for node in nodes_in_cluster])

        modularity += l_c/m - (d_c/(2*m))**2

    return modularity


def spectral_clustering(G, k): # Main spectral clustering algorithm
    # k: number of clusters
    n = G.number_of_nodes()
    A = nx.to_numpy_matrix(G)
    D = np.zeros((n,n))
    for i in range(n):
        D[i,i] = np.sum(A[i,:])

    L = D - A # Construct Laplacian

    eigvals, eigvecs = eigs(L, k=k, which='SR')
    eigvecs = eigvecs.real

    # Kmeans on eigenvectors of Laplacian
    km = KMeans(n_clusters=k)
    km.fit(eigvecs)

    clustering = dict()
    for i, node in enumerate(G.nodes()):
        clustering[node] = km.labels_[i]

    return clustering


def process_spectral_clustering(G, n, name, color, flag='kcore'):
    '''Perform spectral clustering & plot clusters
        G = graph
        n = number of clusters
        name = handle
        flag = corpus, kcore or ktruss
        Computes modularity of clusters and compare to modularity of random clustering
    '''
    verbose = False

    print(f'\nSplitting the {G.number_of_nodes()}-node {flag} graph into {n} spectral custers:')
    clustering_result = spectral_clustering(G, n)
    if verbose: print(clustering_result)
    clusters = set(clustering_result.values())

    for cluster in clusters:
        icluster = cluster + 1 # for display
        nodes_in_cluster = [node for node in G.nodes() if clustering_result[node]==cluster]
        #print(f'Nodes in cluster {cluster}:\n{nodes_in_cluster}')
        subG = G.subgraph(nodes_in_cluster)
        print(f'cluster {icluster}: {subG.number_of_nodes()} nodes / {subG.number_of_edges()} edges')
        prefix = flag + ' ' + 'cluster ' + str(icluster) +':' + str(n)
        suffix = str(subG.number_of_nodes()) + ' nodes'
        title = build_title(prefix, name, suffix)

        # if flag != 'corpus': # pgv plot cpu-intensive
        #     pgv_plot(subG, title, color, plot_directory)
        # else:
        plot_gow(subG, title, color, 'spring', True)

    print(f'Modularity of spectral clustering: {modularity(G, clustering_result):.3g}')
    random_clustering_result = random_clustering(G)
    print(f'Modularity of random clustering: {modularity(G, random_clustering_result):.3g}')