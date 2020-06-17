#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 15:20:01 2020

cloud_of_words.py

@author: charly
"""

# import numpy as np
# import pandas as pd
# from os import path
# from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from tweet_io import load_processed_data

if __name__ == '__main__':
    corpus, labels = load_processed_data()

    print(f'length corpus = {len(corpus)}')
    label_set = list(set(labels))
    print(label_set)
    nlabels = len(label_set)
    print(f'Number of unique labels = {nlabels}')

    word_soups = [''] * nlabels

    for i, tweet in enumerate(corpus):
        for j, label in enumerate(label_set):
            if labels[i] == label_set[j]:
                word_soups[j] += tweet + ' '
                break;

    for i, potage in enumerate(word_soups):
        wordcloud = WordCloud(background_color="white").generate(potage)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        filename = f'word_soup_{label_set[i]}.png'
        wordcloud.to_file(filename)

    text = " ".join(potage for potage in word_soups)

    wordcloud = WordCloud(background_color="white").generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    filename = 'word_soup.png'
    wordcloud.to_file(filename)
