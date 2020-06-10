#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 20:14:07 2020

handles.py

Manages handles & names

@author: charly
"""

def get_handles():
    handles = ['EmmanuelMacron', 'MLP_officiel', 'JLMelenchon']
    return handles


def get_name_dict(anonymous = True):
    '''Anonymize names by default'''
    handles = get_handles()
    if anonymous:
        name_dict = {
            handles[0]: 'Handle A',
            handles[1]: 'Handle B',
            handles[2]: 'Handle C',
            }
    else:
        name_dict = {
            handles[0]: 'Macron',
            handles[1]: 'le Pen',
            handles[2]: 'Mélenchon',
            }
    return name_dict


def get_color_dict():
    handles = get_handles()
    color_dict = {
        handles[0]: 'darkseagreen', # Macron
        handles[1]: 'steelblue', # MLP
        handles[2]: 'tomato', # Mélenchon
        }
    return color_dict



def get_names():
    name_dict  = get_name_dict()
    handles    = get_handles()
    names      = list()

    for handle in handles:
        names.append(name_dict[handle])
    return names