#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 08:10:36 2020

plot_helper.py

functionality useable by all plots

@author: charles mégnin
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tweet_io import clean_filename
from handles import get_name_dict

plot_directory = './plots/gow/'


def build_title(prefix, names='', suffix=''):
    """
    Builds plot title from three elements :
        prefix - typically graph descriptor
        name   - typically twitter handle (optional)
        suffix - additional info (optional)
    adds a space after the prefix and after the name & places parentheses around the suffix
    The title returned is of the form :
        prefix name (suffix)

    eg: prefix = '17-truss'
        names  = 'Mélenchon'
        suffix = 63 nodes window=5
        yields : 17-truss Mélenchon (63 nodes window=5)
    """
    title  = prefix + ' '
    if names != '':
        title += names + ' '
    if suffix != '':
        title += '(' + suffix +')'

    return title

def kernel_to_heatmap(K, title, max_scale, save=True):

    # Generate a mask for the upper triangle
    #mask = np.triu(np.ones_like(K, dtype=np.bool))
    # Generate a custom diverging colormap
    cmap = "YlGn"
    rotation = 0
    x = [0.5, 1.5, 2.5]
    name_dict = get_name_dict()
    handles = name_dict.values()

# Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(K,
                #mask=mask,
                cmap=cmap,
                vmax=max_scale,
                #center=0,
                square=True,
                linewidths=.75,
                cbar_kws={'shrink': .75,
                          'label': 'Kernel values'},
                xticklabels=True,
                yticklabels=True,
                annot=True,)
    sns.set(font_scale=2)
    sns.set(style="white")

    plt.xticks(x, handles, rotation=rotation)
    plt.yticks(x, handles, rotation=0)
    plt.title(title)
    if save:
        filename = clean_filename(title, 'png', plot_directory)
        print(f'saving plot to {filename}')
        plt.savefig(filename)
    plt.show()
