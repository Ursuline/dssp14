#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 21:45:35 2020

graph_of_words.py

Move to more appropriate file

@author: charly
"""
import networkx as nx
import unidecode
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from tweet_io import load_data_by_name
from tweet_processor import preprocess
from handles import get_color_dict

plot_dir = './plots/gow/'

def create_graph_of_words(tweet, window_size, directed = False):
    # Generates a graph of words for a given tweet
    if directed == True:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    tokens = tweet.split()
    n = len(tokens)

    for i in range(n):
        for j in range(i+1,i+window_size):
            if j < n and tokens[i] != tokens[j]:
                G.add_edge(tokens[i], tokens[j])

    # Add labels as 'label'
    for node in G.nodes():
        G.nodes[node]["label"] = node

    #print(f"{G.number_of_nodes()} nodes ", end='')
    #print(f"{G.number_of_edges()} edges")

    return G


def plot_graph_of_words(G, handle, title=''):
    """
    Plots graph of words
    Wrapper around nx.draw_network
    """
    size       = 15
    width      = .15
    node_size  = 200
    font_size  = 30
    title_size = 25

    directory = './plots/'
    title     = title + '_' + handle
    filename  = directory + 'GOW_'+ title + '.png'
    filename  = unidecode.unidecode(filename) # remove accents
    color_dict = get_color_dict()

    font_color = color_dict[handle]
    node_color = 'grey'

    plt.figure(figsize=(size, size))
    plt.title(title, fontsize=title_size)
    nx.draw_networkx(G,
                     width      = width,
                     node_size  = node_size,
                     font_size  = font_size,
                     font_color = font_color,
                     node_color = node_color)
    plt.savefig(filename)
    plt.show()


def gow(processed_data):
    """
    Create the graph-of-words representation of each tweet,
    and append it to the Gs list.
    Set the size of the sliding window equal to 3.
    """
    Gs = list()
    window_size = 3

    for doc in processed_data:
        G = create_graph_of_words(doc, window_size, True)
        Gs.append(G)

    #We will next extract some features from our set of graphs and we will use these features to perform classification. Specifically, we will use the following three features: (i) nodes, (ii) paths of length 2, and (iii) triangles (i.e., directed cycles of length 3).

    #The code below extracts all these features from the list of graphs. The features are stored in the features dictionary which maps each feature to a unique integer value. Nodes are represented by their names, paths of length 2 are represented as tuples (node 1, node 2), and triangle are also represented as tuples (node 1, node 2, node 3).

    features = dict()

    for i,G in enumerate(Gs):
        for n1 in G.nodes():
            if n1 not in features:
                features[n1] = len(features)
            for n2 in G.successors(n1):
                if (n1,n2) not in features:
                    features[(n1, n2)] = len(features)
                for n3 in G.successors(n2):
                    if n1 != n3 and G.has_edge(n3, n1):
                        if (n1,n2,n3) not in features:
                            features[(n1, n2, n3)] = len(features)

    X = lil_matrix((len(Gs), len(features)))

    # Generate document-feature matrix
    for i,G in enumerate(Gs):
        for n1 in G.nodes():
            X[i,features[n1]] += 1
            for n2 in G.successors(n1):
                X[i,features[(n1, n2)]] += 1
                for n3 in G.successors(n2):
                    if n1 != n3 and G.has_edge(n3, n1):
                        X[i,features[(n1, n2, n3)]] += 1

    print("Shape of data matrix:", X.shape)
    return X

if __name__ == '__main__':

    # handle = 'EmmanuelMacron'

    # raw_tweets, class_labels = load_data_by_name(handle)

    # stemmer_flag = False # set to True to perform stemming
    # cv=CountVectorizer()
    # word_count_vector=cv.fit_transform(raw_tweets)
    # print("Word count before processing:")
    # print(word_count_vector.shape)

    # processed_tweets, class_labels = preprocess(raw_tweets,
    #                                             class_labels,
    #                                             stemmer_flag)

    # n_tweets = 104
    # filename = handle + '_' + str(n_tweets) + '_processed.txt'

    # with open(filename, 'w') as f:
    #     f.writelines("%s\n" % tweet for tweet in processed_tweets[:n_tweets])

    # X = gow(processed_tweets)


    tweets = ['There is nothing but', 'fear but life itself']
    X = gow(tweets)
    print(X.todense())