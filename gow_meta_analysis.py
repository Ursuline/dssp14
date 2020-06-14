#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 10:10:50 2020

gow_meta_analysis.py

@author: charles mégnin
"""
print(__doc__)

import time
import numpy as np
import matplotlib.pyplot as plt

from handles import get_handles, get_names, get_name_dict, get_color_dict
from tweet_io import clean_filename

plot_directory = './plots/gow/'

def plot_Gmeta(array1, array2, array3, title, ylabel, density):
    barWidth = .3
    colors = ['tan', 'yellowgreen', 'midnightblue']
    labels = ['Corpus', 'k-core', 'k-truss']
    names  = get_names()

    bars = [array1, array2, array3]

    fig, ax = plt.subplots()
    # Set position of bar on X axis
    r = list()
    r.append(np.arange(len(bars[0])) + barWidth/2)
    r.append([x + barWidth for x in r[0]])
    r.append([x + barWidth for x in r[1]])

    # Make the plot
    for i in range(3):
        plt.bar(r[i],
                bars[i],
                color=colors[i],
                width=barWidth,
                edgecolor='white',
                label=labels[i])

    text_x_offset = .1
    text_y_offset = 1.5
    for i in range(3):
        for j in range(3):
            x_pos = r[i][j] - barWidth/3 + text_x_offset
            if density == False:
                y_pos = bars[i][j] + np.exp(text_y_offset)
                label = f'{bars[i][j]:.0f}'
            else:
                y_pos = bars[i][j] + np.log(1.0001)
                label = f'{bars[i][j]:.3f}'
            plt.text(x_pos,
                      y_pos,
                      label,
                      horizontalalignment='center',
                      rotation=0,
                      color='black',
                      fontsize=10)

    # Add ticks on the middle of the group bars
    plt.ylabel(ylabel, fontweight='normal')
    plt.xticks([r + barWidth for r in range(len(array1))], names)
    ax.set_yscale('log')
    plt.title(title)

    # Create legend show graphic & save to file
    plt.legend(loc='best', fontsize='small')
    plt.savefig(clean_filename(title , 'png', plot_directory))
    plt.show()


if __name__ == '__main__':
    t0 = time.time()

    name_dict  = get_name_dict()
    color_dict = get_color_dict()

    Gcorpus_nodes = np.array([7350, 8872, 7249])
    Gcorpus_edges = np.array([81032, 103925, 81362])
    Gcore_nodes   = np.array([271, 280, 229])
    Gcore_edges   = np.array([8007, 8672, 6847])
    Gtruss_nodes  = np.array([59, 63, 63])
    Gtruss_edges  = np.array([996, 1153, 1249])

    Gcorpus_sparcity = np.zeros(3)
    Gcore_sparcity   = np.zeros(3)
    Gtruss_sparcity  = np.zeros(3)

    for i, handle in enumerate(get_handles()):
        nodes = Gcorpus_nodes[i]
        edges = Gcorpus_edges[i]
        Gcorpus_sparcity[i] = edges /(.5*nodes*(nodes-1))
        nodes = Gcore_nodes[i]
        edges = Gcore_edges[i]
        Gcore_sparcity[i] = edges /(.5*nodes*(nodes-1))
        nodes = Gtruss_nodes[i]
        edges = Gtruss_edges[i]
        Gtruss_sparcity[i] = edges /(.5*nodes*(nodes-1))

    plot_Gmeta(Gcorpus_nodes,
               Gcore_nodes,
               Gtruss_nodes,
               'Order of dense subgraphs',
               '# vertices (log)', False)

    plot_Gmeta(Gcorpus_edges,
               Gcore_edges,
               Gtruss_edges,
               'Size of dense subgraphs',
               '# edges (log)',False)

    plot_Gmeta(Gcorpus_sparcity,
               Gcore_sparcity,
               Gtruss_sparcity,
               'Density of dense subgraphs',
               'density (log)', True)