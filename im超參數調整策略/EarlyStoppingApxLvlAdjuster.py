# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 09:49:07 2024

@author: user
"""

import numpy as np

class EarlyStoppingApxLvlAdjuster:
    def __init__(self, start=0.1, end=1.0, patience=10):
        self.start = start
        self.end = end
        self.patience = patience
        self.best_performance = float('-inf')
        self.epochs_without_improvement = 0
        self.apxLvl = start

    def adjust(self, val_performance):
        if val_performance > self.best_performance:
            self.best_performance = val_performance
            self.epochs_without_improvement = 0
            # 性能提升，增加 apxLvl
            self.apxLvl = min(self.end, self.apxLvl + 0.01)
        else:
            self.epochs_without_improvement += 1
            # 性能未提升，略微降低 apxLvl
            self.apxLvl = max(self.start, self.apxLvl - 0.005)

        # 根據距離 early stopping 的接近程度調整 apxLvl
        proximity_to_stopping = self.epochs_without_improvement / self.patience
        self.apxLvl = min(self.end, self.apxLvl + proximity_to_stopping * 0.2)

        return self.apxLvl, self.epochs_without_improvement >= self.patience

# 使用示例
adjuster = EarlyStoppingApxLvlAdjuster()

for epoch in range(200):  # 假設最多訓練 200 個 epoch
    # 模擬訓練和驗證
    val_performance = np.random.rand()  # 在實際應用中，這裡應該是真實的驗證性能
    
    apxLvl, should_stop = adjuster.adjust(val_performance)
    
    print(f"Epoch {epoch}, ApxLvl: {apxLvl:.4f}, Val Performance: {val_performance:.4f}")
    
    if should_stop:
        print(f"Early stopping at epoch {epoch}")
        break