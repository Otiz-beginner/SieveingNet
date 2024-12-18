# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 21:57:31 2024

@author: user
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_overlapping_bar_chart(count_dict_a, count_dict_b, figsize=(12, 7)):
    keys = list(set(count_dict_a.keys()) | set(count_dict_b.keys()))
    keys.sort()
    
    values_a = [count_dict_a.get(k, 0) for k in keys]
    values_b = [count_dict_b.get(k, 0) for k in keys]
    
    x = np.arange(len(keys))
    width = 0.5
    
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot bars for count_dict_a
    bars1 = ax1.bar(x, values_a, width, label='Quantities', color='#4E79A7', edgecolor='black')
    
    # Plot bars for count_dict_b
    bars2 = ax1.bar(x, values_b, width, label='Impact', color='#F28E2B', edgecolor='black', alpha=0.7)
    
    ax1.set_xlabel('Exponent value', fontsize=12)
    ax1.set_ylabel('Count / Impact', fontsize=12)
    ax1.set_title('Quantities and Impact of different exponent sizes', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(keys)
    
    # Improve the appearance of the plot
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.legend()
    
    plt.tight_layout()
    
    return fig, ax1

count_dict_a = {1: 100, 2: 200, 3: 150, 4: 300}
count_dict_b = {1: 50, 2: 180, 3: 120, 4: 220}
fig, ax = plot_overlapping_bar_chart(count_dict_a, count_dict_b)
plt.show()