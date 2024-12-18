# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 22:06:00 2024

@author: user
"""

import numpy as np
import time

def float_to_fixed_point_binary(arr, int_bits=4, frac_bits=12):
    # 計算轉換係數
    scale = 2 ** frac_bits
    
    # 將浮點數乘以係數
    scaled_arr = np.round(arr * scale).astype(np.int32)
    
    # 計算整數的最大值和最小值
    max_val = (2 ** (int_bits + frac_bits)) - 1
    min_val = -(2 ** (int_bits + frac_bits))
    
    # 確保數值在允許的範圍內
    scaled_arr = np.clip(scaled_arr, min_val, max_val)
    
    def to_binary(val):
        if val >= 0:
            # 對於正數，直接轉成binary
            return np.unpackbits(np.array([val], dtype='>u2').view(np.uint8))[-(int_bits + frac_bits):]
            # >：Big-endian，u：Unsigned，2：two words（16bits），4: four words（32bits）
        else:
            # 對於正數，轉成binary最後再加上 - 號
            val = -val  # 先取絕對值
            bits = np.unpackbits(np.array([val], dtype='>u2').view(np.uint8))[-(int_bits + frac_bits):]
            return -bits
    
    # 將每個元素轉換為二進制
    binary_matrix = np.array([to_binary(val) for val in scaled_arr.flatten()])
    
    # 重塑為原始形狀並增加額外的維度
    result_shape = arr.shape + (int_bits + frac_bits,)
    binary_matrix = binary_matrix.reshape(result_shape)
    
    return binary_matrix.astype(np.int8)

# 測試範例
input_array = np.array([[4.625, 0.02500], [-0.255, -1.05]])
start_time = time.time()
for t in range(1):
    binary_result = float_to_fixed_point_binary(input_array, int_bits=4, frac_bits=12)
end_time = time.time()

print(binary_result)

print(f"total time: {end_time - start_time}")

print(f"type(binary_result): {type(binary_result[0, 0, 0])}")

