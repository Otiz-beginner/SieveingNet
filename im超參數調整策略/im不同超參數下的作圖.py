# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 22:40:55 2024

@author: Otiz
"""

import numpy as np
import matplotlib.pyplot as plt

def im_function(current_epoch, total_epochs, im_start, im_end, k):
    return im_start + (im_end - im_start) * (1 - np.exp(-k * current_epoch / total_epochs))

total_epochs = 10
im_start = 0.1
im_end = 1
k_values = [3, 5, 7]
colors = ['red', 'green', 'blue']

epochs = np.linspace(0, total_epochs, 100)

plt.figure(figsize=(10, 6))

for k, color in zip(k_values, colors):
    im_values = im_function(epochs, total_epochs, im_start, im_end, k)
    plt.plot(epochs, im_values, color=color, label=f'k={k}')

plt.xlabel('Epoch')
plt.ylabel('im')
plt.title('im Function for Different k Values')
plt.legend()
plt.grid(True)
plt.show()