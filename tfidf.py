#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:23:50 2020

@author: charles mégnin
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

VERBOSE = 0

def tf_idf(data):
    """
    Compute TF-IDF
    return transformed data
    """
    print(f"[tf_idf] Input data shape: {np.shape(data)}")
    cv=CountVectorizer()
    # generate word count for the words in data
    word_count_vector=cv.fit_transform(data)
    print(f"[tf_idf] Word count after processing: {word_count_vector.shape}")

    vec = TfidfVectorizer(ngram_range=(1, 2)) # set to 1, 2 for bigrams

    X = vec.fit_transform(data)
    print(f'[tf_idf]: TF-IDF vector shape: {X.shape}')

    # size is # of non-zero values in matrix
    if VERBOSE == 1:
        with open('data-tfidx.txt', 'w') as f:
            for item in data:
                f.write("%s\n" % item)
        print(f"Sparsity: {1 - X.size / (X.shape[1] * X.shape[0]):.2g}")
        print("*** TF-IDF ***")
        print(vec.vocabulary_.keys()) # index
        #print(vec.vocabulary_)
        print(X)
        print("*** END TF-IDF ***")

    return X
