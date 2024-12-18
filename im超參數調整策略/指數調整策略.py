# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 19:08:16 2024

@author: user
"""

import matplotlib.pyplot as plt
import numpy as np
# 假設這裡是你計算得到的 loss 資料，這裡我們使用隨機生成的示例資料
current_epoch = np.arange(1, 11, 1)  # 生成 100 個從 0.1 到 10 的等間隔數值作為示例
print(f"current_epoch: {current_epoch}")
# 定義變量
im_start = 0.001  # 替換為您的起始值
im_end = 1.0    # 替換為您的結束值
total_epochs = 10  # 替換為總的 epoch 數

# 計算公式
im = im_end - (im_end - im_start) * np.exp(-3 * current_epoch / total_epochs)

print(f"im: {im}")

# delta_e = (1 / -accuracy) * (1 / np.log(10))

plt.figure(figsize=(10, 6))
plt.plot(current_epoch, im, marker='o', linestyle='-', color='b', label='e vs loss')
plt.title('im and epoch')
plt.xlabel('epoch')
plt.ylabel('im')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()