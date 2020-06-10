#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 09:04:51 2020

gow_plots.py

plots for gow_analysis

@author: charly
"""
import networkx as nx
import matplotlib.pyplot as plt

from pgv_helper import pgv_plot
from tweet_io import clean_filename
from plot_helper import build_title

plot_directory = './plots/gow/'
data_directory = './gml/'

def get_layout_dict():
    layout_dict = {
        'corpus': 'spring',
        'max_clique': 'spring',
        'commonality': 'circular',
        'unicity': 'circular'
        }
    return layout_dict


def map_layouts():
    '''return layout options for networkx plots'''
    layouts = {
        'circular': nx.circular_layout,
        'shell': nx.shell_layout,
        'spring': nx.spring_layout,
        'spectral': nx.spectral_layout,
        'random': nx.random_layout,
        }

    return layouts


def plot_gow(G, title, color, layout, save=True):
    """
    Plots graph of words
    Wrapper around nx.draw_network
    """
    verbose    = True
    size       = 15
    width      = .15
    node_size  = 200
    font_size  = 30
    title_size = 25
    node_color = 'grey'

    layouts = map_layouts()
    pos     = layouts[layout](G)

    plt.figure(figsize=(size, size))
    plt.title(title, fontsize=title_size)
    nx.draw_networkx(G,
                     pos,
                     width      = width,
                     node_size  = node_size,
                     node_color = node_color,
                     font_size  = font_size,
                     font_color = color,)
    if save:
        if verbose: print(f'saving plot to {clean_filename(title , "png", plot_directory)}')
        plt.savefig(clean_filename(title , 'png', plot_directory))
    plt.show()


def plot_commonality(Gr, name_dict, handles, window, color, dense_flag, layout, exclusive = True):
    """
    Plots graph Gr computed from common nodes/edges to two graphs
    exclusive: common to the two graphs only. If False, common also with other graph(s)
    """
    if dense_flag == 'kcore': dense_flag = 'main core'

    prefix = 'Common nodes ' + dense_flag
    if exclusive==False: prefix += ' inclusive'
    if len(handles) == 3:
        name = 'all handles'
    else:
        name = name_dict[handles[0]] +'-' + name_dict[handles[1]]
    suffix = str(nx.number_of_nodes(Gr)) + ' nodes'
    #suffix += ' window=' + str(window)
    title = build_title(prefix, name, suffix)

    plot_gow(Gr, title, color, layout, False)
    if exclusive == True:
        pgv_plot(Gr, title, color, plot_directory)


def plot_unicity(Sp, name_dict, handle, dense_flag, window, layout, color):
    if dense_flag == 'kcore': dense_flag = 'main core'

    prefix = 'Unique nodes ' + dense_flag
    suffix = str(nx.number_of_nodes(Sp)) + ' nodes'
    #suffix += ' window=' + str(window)
    name   = name_dict[handle]
    title = build_title(prefix, name, suffix)

    plot_gow(Sp, title, color, layout, False)
    pgv_plot(Sp, title, color, plot_directory)


def plot_max_clique(G_max, name_dict, handle, graph_type, color, layout):
    print(f'max clique {graph_type}:')

    prefix = 'Max clique ' + graph_type
    name   = name_dict[handle]
    suffix = str(G_max.number_of_nodes()) + ' nodes'
    title  = build_title(prefix, name, suffix)

    # only save pgv plot
    plot_gow(G_max, title, color, layout, False)
    if graph_type != 'corpus':
        pgv_plot(G_max, title, color, plot_directory)