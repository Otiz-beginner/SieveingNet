# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 17:50:34 2024

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt

def count_elements(arr):
    unique_elements, counts = np.unique(arr, return_counts=True)
    result_dict = dict(zip(unique_elements, counts))
    return result_dict

def exponent_normalization(input_dict):
    max_key = max(input_dict.keys())  # 確保我們用的是正確的最大值
    normalized_dict = {}
    for element, counts in input_dict.items():
        normalized_dict[element] = counts / (2 ** (max_key - element))
    return normalized_dict

def plot_exponent_impact(count_dict):
    keys = list(count_dict.keys())
    values = list(count_dict.values())
    
    plt.bar(keys, values)
    plt.xlabel('Exponent value')
    plt.ylabel('Impact')
    plt.title('Impact of different exponent sizes')
    plt.show()
    
def plot_exponent_counts(count_dict):
    keys = list(count_dict.keys())
    values = list(count_dict.values())
    
    plt.bar(keys, values)
    plt.xlabel('Exponent value')
    plt.ylabel('Count')
    plt.title('Quantities of different exponent sizes')
    plt.show()

# 測試
arr = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
count_dict = count_elements(arr)
normalized_dict = exponent_normalization(count_dict)
plot_exponent_counts(normalized_dict)


