#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:42:44 2020

pgv_helper.py

Helper routines for pygraphviz plots

@author: charles mégnin
"""

import networkx as nx
#import pygraphviz as pgv

from tweet_io import clean_filename

def _init_graph(G_nx, title, node_color, font_size):
    '''Initialize pygraphviz graph from nx Graph'''
    bg_color = 'white'
    title_size = 25
    node_shape = 'box'
    node_style = 'rounded, filled'
    box_color = 'black' # set to bgcolor to make it disappear
    font_color = 'black' # node font color
    node_width = 2.5 # node_shape width
    eedge_color = 'dimgrey'

    print(f'Initializing graph with title="{title}", node color={node_color}, font size={font_size}')

    Ga = nx.drawing.nx_agraph.to_agraph(G_nx)
    Ga.graph_attr.update({
        'bgcolor': bg_color,
        'penwidth': 5,
        'label': title,
        'fontsize': title_size,
        #'splines': edge_splines,
        }
    )
    Ga.node_attr.update({
        'shape': node_shape,
        'style': node_style,
        'fixedsize': True,
        'color': box_color,
        'fontcolor': font_color,
        'fillcolor': node_color,
        'fontsize': 30,
        'width': node_width,
        }
    )
    Ga.edge_attr.update({
        #'fontsize': font_size,
        'shape': 'normal',
        'color': eedge_color,
        }
    )
    return Ga

def pgv_plot(G, prefix, color, directory):
    """pygraphviz graph vizualisation"""
    verbose = 0

    # Convert nx graph to pyvizgraph
    A = _init_graph(G, prefix, color, 20)

    filename = clean_filename(prefix + '_pgv', 'png', directory)
    if verbose: print(f'Saving plot as {filename}')

    A.draw(filename, prog="circo")
