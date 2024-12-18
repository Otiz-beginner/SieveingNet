# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 17:07:55 2024

@author: Otiz
"""

import matplotlib.pyplot as plt
import numpy as np
# 假設這裡是你計算得到的 loss 資料，這裡我們使用隨機生成的示例資料
accuracy = np.linspace(0, 1, 100)  # 生成 100 個從 0.1 到 10 的等間隔數值作為示例
print(accuracy)
# accuracy = np.array([0.10, 0.30, 0.31, 0.89, 0.90, 1.0])
loss = np.linspace(0, 1, 1000)
# 計算對應的 delta_e
e = np.log10(1.001-accuracy)
# delta_e = (1 / -accuracy) * (1 / np.log(10))

plt.figure(figsize=(10, 6))
plt.plot(accuracy, e, marker='o', linestyle='-', color='r', label='e vs loss')
plt.title('Relationship between delta_e and accuracy')
plt.xlabel('accuracy')
plt.ylabel('e')
plt.grid(True)
plt.legend()
plt.tight_layout()
# 顯示圖形
plt.show()
# print(f"e: {e}")
print(f"Early stage: {e[2] - e[1]}")
print(f"Late stage: {e[4] - e[3]}")

# plt.figure(figsize=(10, 6))
# plt.plot(accuracy, accuracy, marker='o', linestyle='-', color='b', label='e vs loss')
# plt.title('Relationship between delta_e and accuracy')
# plt.xlabel('accuracy')
# plt.ylabel('e')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()

# # 繪製圖形
# plt.figure(figsize=(10, 6))
# plt.plot(accuracy, delta_e, marker='o', linestyle='-', color='b', label='delta_e vs loss')
# plt.title('Relationship between delta_e and accuracy')
# plt.xlabel('accuracy')
# plt.ylabel('delta_e')
# plt.grid(True)
# plt.legend()
# plt.show()