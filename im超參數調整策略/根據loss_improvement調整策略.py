# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 00:59:25 2024

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
# Given list
values = [0.8330553869187247, 0.3703817369272457, 0.2879088352402839, 0.24332111465278802,
          0.2125925089038988, 0.18573559156730413, 0.1673070258520667, 0.15107035737803254,
          0.13656687312105562, 0.1244413006953059]

# Calculate the differences between each neighbor
differences = -np.diff(values)
print(differences)
# 假設這裡是你計算得到的 loss 資料，這裡我們使用隨機生成的示例資料
improvement = np.array([0.46267365, 0.0824729, 0.04458772, 0.03072861, 0.02685692, 0.01842857, 0.01623667, 0.01450348, 0.01212557])

# 計算公式
im = 1 - np.exp(-1 * improvement)

print(f"im: {im}")

# delta_e = (1 / -accuracy) * (1 / np.log(10))

plt.figure(figsize=(10, 6))
plt.plot(improvement, im, marker='o', linestyle='-', color='b', label='e vs loss')
plt.title('im and improvement')
plt.xlabel('improvement')
plt.ylabel('im')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()