#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 16:13:46 2020

plots_eda.py

plots for tweets_eda

@author: charles mégnin
"""
import numpy as np
import datetime
import matplotlib.pyplot as plt

from handles import get_names
from plot_helper import build_title, clean_filename

plot_directory = './plots/eda/'

def get_ndays(start, end):
    '''
    return number of days between dates start & end
    start, end are arrays of stringin the format %d/%m/%Y
    '''
    ndays = np.zeros(len(start))

    for i in range(len(start)):
        date1 = datetime.datetime.strptime(start[i], '%d/%m/%Y').date()
        date2 = datetime.datetime.strptime(end[i], '%d/%m/%Y').date()
        ndays[i] = (date2-date1).days
    #print(f'# days:{ndays}')

    return ndays


def build_base_barplot(handles, array, x_pos, title, xlabel, ylabel, color_dict):
    '''Builds base plots for 3 handles'''
    width = .75
    names = get_names()

    fig, ax = plt.subplots()

    barlist=ax.bar(x_pos, array, width)

    for i, handle in enumerate(handles):
        barlist[i].set_color(color_dict[handle])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(names)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title = title
    ax.set_title(title)

    return fig, ax

def plot_distribution(dic, name, prefix, xlabel, max_x, max_y, color, mu, sigma):
    width = .85
    annot_color = 'saddlebrown'
    X = dic.keys()
    Y = dic.values()

    plt.bar(X, Y, width, color=color)
    plt.axis([0, max_x, 0, max_y])

    # build vertical bar for mean with legend
    plt.axvline(x=mu, color = annot_color, linestyle='--')
    annot  = rf'$\mu={mu:.1f}$ $\sigma={sigma:.1f}$'

    plt.text(mu+.25, 0.95*max_y, annot, rotation=0, color=annot_color)

    # build title and axes labels
    title = build_title(prefix, name, '')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('count')

    plt.savefig(clean_filename(title , 'png', plot_directory))
    plt.show()

def tweet_count_plot(bars1, bars2, handles):
    barWidth = 0.4 # set width of bar
    colors = ['tan', 'yellowgreen']
    labels = ['Raw tweets', 'Processed']

    bars = [bars1, bars2]
    arr = np.array(bars)

    # Set position of bar on X axis
    r = list()
    r.append(np.arange(len(bars[0])) + barWidth/2)
    r.append([x + barWidth for x in r[0]])

    # Make the plot
    for i in range(2):
        plt.bar(r[i],
                bars[i],
                color=colors[i],
                width=barWidth,
                edgecolor='white',
                label=labels[i])

    text_x_offset = .06
    text_y_offset = 250
    for i in range(2):
        for j in range(3):
            plt.text(r[i][j]-barWidth/2 + text_x_offset,
                     bars[i][j] - text_y_offset,
                     f'{bars[i][j]/np.sum(arr, axis=1)[i]*100:.1f}%',
                     rotation=0,
                     color='black',
                     fontsize=10)

    # Add ticks on the middle of the group bars
    plt.ylabel('# tweets', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))], handles)
    title = 'Tweet count'
    plt.title(title)

    # Create legend show graphic & save to file
    plt.legend()
    title = 'Tweet pre-processing'
    plt.savefig(clean_filename(title , 'png', plot_directory))
    plt.show()

def tweeting_period_plot(handles, start, end, color_dict):
    '''Plot tweeting period from start and end dates'''
    fontsize = 14

    names = get_names()
    ndays = get_ndays(start, end)

    fig, ax = plt.subplots(figsize=(18,4.875))

    y_pos = np.arange(len(handles))
    barlist=ax.barh(y_pos, ndays, .75)

    for i, handle in enumerate(handles):
        barlist[i].set_color(color_dict[handle])

    # Annotate bars
    end_x   = .89
    start_x = [.05, .275, .55]
    y_val   = [.815, .485, .15]
    for i, handle in enumerate(handles):
        ax.annotate('<-- ' + start[i],
                    xy=(start_x[i], y_val[i]),
                    xycoords = 'axes fraction',
                    fontsize = fontsize)
        ax.annotate(end[i] + ' -->',
                    xy=(end_x, y_val[i]),
                    xycoords = 'axes fraction',
                    fontsize = fontsize)

    plt.gca().invert_xaxis() # right to left
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('# of days')
    title = 'Tweeting time period'
    ax.set_title(title, fontsize = 18)
    plt.savefig(clean_filename(title , 'png', plot_directory))

    plt.show()


def tweet_frequency_plot(handles, start, end, ntweets, color_dict):
    x_offset = .075 # offsets wrt x & y position for bar legends
    y_offset = .5
    annot_color = 'saddlebrown'
    fontsize = 12
    x_pos = np.arange(len(handles))

    ndays = get_ndays(start, end)
    title = 'Tweet frequency'

    freq = list()
    mu = 0 # compute mean frequency
    for i, handle in enumerate(handles):
        freq.append(ntweets[i]/ndays[i])
        mu += ntweets[i]
    mu /= np.sum(ndays)
    print(f'Mean frequency={mu}')

    fig, ax = build_base_barplot(handles, freq, x_pos, title, '', '# tweets/day', color_dict)

    for j in range(3):
        ax.text(x_pos[j] - x_offset,
                 freq[j] - y_offset,
                 f'{freq[j]:.1f}',
                 rotation=0,
                 color='black',
                 fontsize=fontsize)

    # build vertical bar for mean with legend
    plt.axhline(y=mu, color = annot_color, linestyle='--')
    annot = rf'$\mu$=' + str(f"{mu:.1f}")

    plt.text(-.4, mu+.2, annot, rotation=0, color=annot_color)

    plt.savefig(clean_filename(title , 'png', plot_directory))

    plt.show()

def word_volume_plot(handles, wc, color_dict):
    '''
    plot word count
    wc is list of dictionaries
    '''
    x_offset = .15 # offsets wrt x & y position for bar legends
    y_offset = 3000
    fontsize = 10
    x_pos = np.arange(len(handles))
    title = 'Word volume'

    word_count = np.zeros(len(handles))
    for i, handle in enumerate(handles):
        for key, value in wc[i].items():
            word_count[i] += int(key*value)
        print(f'word count {handles[i]}={int(word_count[i]):d}')

    fig, ax = build_base_barplot(handles, word_count, x_pos, title, '', '# words', color_dict)

    for j in range(3):
        ax.text(x_pos[j] - x_offset,
                 word_count[j] - y_offset,
                 f'{int(word_count[j]):d}',
                 rotation=0,
                 color='black',
                 fontsize=fontsize)
    plt.savefig(clean_filename(title , 'png', plot_directory))
    plt.show()

def unique_word_volume_plot(handles, wc, color_dict):
    '''
    plot unique word count
    wc is list of dictionaries
    '''
    x_offset = .125 # offsets wrt x & y position for bar legends
    y_offset = 500
    fontsize = 10
    x_pos = np.arange(len(handles))
    title = 'Unique word volume'

    word_count = np.zeros(len(handles))
    for i, handle in enumerate(handles):
        word_count[i] = int(wc[i])
    print(f'word count {handles[i]}={int(word_count[i]):d}')

    fig, ax = build_base_barplot(handles, word_count, x_pos, title, '', '# words', color_dict)

    for j in range(3):
        ax.text(x_pos[j] - x_offset,
                 word_count[j] - y_offset,
                 f'{int(word_count[j]):d}',
                 rotation=0,
                 color='black',
                 fontsize=fontsize)
    plt.savefig(clean_filename(title , 'png', plot_directory))
    plt.show()

def diversity_plot(handles, div, color_dict):
    '''
    plot unique words / total words
    wc is list
    '''
    x_offset = .1 # offsets wrt x & y position for bar legends
    y_offset = .025
    fontsize = 10
    x_pos = np.arange(len(handles))
    title = 'Diversity of vocabulary'

    word_count = np.zeros(len(handles))
    for i, handle in enumerate(handles):
        word_count[i] = div[i]
    print(f'[diversity_plot]: diversity {handles[i]}={int(word_count[i]):.2f}')

    fig, ax = build_base_barplot(handles, word_count, x_pos, title, '', 'unique words / words', color_dict)

    for j in range(3):
        ax.text(x_pos[j] - x_offset,
                 word_count[j] - y_offset,
                 f'{word_count[j]:.2f}',
                 rotation=0,
                 color='black',
                 fontsize=fontsize)
    plt.savefig(clean_filename(title , 'png', plot_directory))
    plt.show()

def uwpt_plot(handles, uwpt, color_dict):
    '''
    plot unique word count
    wc is list
    '''
    x_offset = .1 # offsets wrt x & y position for bar legends
    y_offset = .5
    fontsize = 10
    x_pos = np.arange(len(handles))
    title = 'Tweet diversity'

    word_count = np.zeros(len(handles))
    for i, handle in enumerate(handles):
        word_count[i] = uwpt[i]
    print(f'[uwpt_plot]: word count {handles[i]}={int(word_count[i]):.2f}')

    fig, ax = build_base_barplot(handles, word_count, x_pos, title, '', 'unique words / tweet', color_dict)

    for j in range(3):
        ax.text(x_pos[j] - x_offset,
                 word_count[j] - y_offset,
                 f'{word_count[j]:.2f}',
                 rotation=0,
                 color='black',
                 fontsize=fontsize)
    plt.savefig(clean_filename(title , 'png', plot_directory))
    plt.show()