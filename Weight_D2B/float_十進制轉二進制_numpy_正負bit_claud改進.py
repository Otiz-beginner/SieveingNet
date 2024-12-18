# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:20:39 2024

@author: user
"""

import numpy as np

def float_to_fixed_point_binary_optimized(arr, int_bits=4, frac_bits=12):
    total_bits = int_bits + frac_bits
    scale = 2 ** frac_bits
    
    # 將浮點數乘以係數並四捨五入
    scaled_arr = np.round(arr * scale).astype(np.int32)
    
    # 計算整數的最大值和最小值
    max_val = (1 << (total_bits - 1)) - 1
    min_val = -(1 << (total_bits - 1))
    
    # 確保數值在允許的範圍內
    np.clip(scaled_arr, min_val, max_val, out=scaled_arr)
    
    # 創建結果陣列
    result = np.zeros(arr.shape + (total_bits,), dtype=np.int8)
    
    # 處理絕對值
    abs_scaled_arr = np.abs(scaled_arr)
    
    # 使用矢量化操作來計算二進制表示
    for i in range(total_bits):
        result[..., -(i+1)] = (abs_scaled_arr >> i) & 1
    
    # 對負數進行符號反轉
    negative_mask = scaled_arr < 0
    result[negative_mask] *= -1
    
    return result

# 測試範例
input_array = np.array([[12.625, 0.02506763], [-0.04475388, 0.07006475]])
binary_result = float_to_fixed_point_binary_optimized(input_array, int_bits=4, frac_bits=12)

print(f"binary_result: {binary_result}")

print(f"type(binary_result): {type(binary_result[0, 0, 0])}")