# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 19:20:47 2024

@author: Otis

用途: 負數為二補數表示
"""

import numpy as np

def float_to_fixed_point_binary(arr, int_bits=1, frac_bits=15):
    # 計算轉換係數
    scale = 2 ** frac_bits
    print(f"arr * scale: {arr * scale}")
    # 將浮點數乘以係數
    scaled_arr = np.round(arr * scale).astype(np.int32)
    
    # 計算整數的最大值和最小值
    max_val = (2 ** (int_bits + frac_bits)) - 1
    min_val = -(2 ** (int_bits + frac_bits))
    print(f"max_val: {max_val}")
    print(f"min_val: {min_val}")
    # 確保數值在允許的範圍內
    scaled_arr = np.clip(scaled_arr, min_val, max_val)
    print(f"arr * scale: {arr * scale}")
    # 處理負數：轉換為二進制表示，並使用2的補數形式處理負數
    def to_binary(val):
        if val >= 0:
            return np.unpackbits(np.array([val], dtype='>u2').view(np.uint8))[-(int_bits + frac_bits):]
            # >：Big-endian，u：Unsigned，2：two words（16bits）
        else:
            # 負數的情況下使用2的補數形式
            val = (1 << (int_bits + frac_bits)) + val
            return np.unpackbits(np.array([val], dtype='>u2').view(np.uint8))[-(int_bits + frac_bits):]
    
    # 將每個元素轉換為二進制
    binary_matrix = np.array([to_binary(val) for val in scaled_arr.flatten()])
    
    # 重塑為原始形狀並增加額外的維度
    result_shape = arr.shape + (int_bits + frac_bits,)
    binary_matrix = binary_matrix.reshape(result_shape)
    
    return binary_matrix.astype(np.int8)

# 測試範例
input_array = np.array([[0.5, 0.02506763], [-0.04475388, 0.07006475]])
binary_result = float_to_fixed_point_binary(input_array)
print(binary_result)
print(f"type(binary_result): {type(binary_result[0, 0, 0])}")
