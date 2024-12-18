# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 23:25:09 2024

@author: user
"""

import numpy as np

def float_to_fixed_point_binary(arr, int_bits=1, frac_bits=15):
    # 計算轉換係數
    scale = 2 ** frac_bits
    
    # 將浮點數乘以係數
    scaled_arr = np.round(arr * scale).astype(np.int32)
    
    # 計算整數的最大值和最小值
    max_val = (2 ** (int_bits + frac_bits)) - 1
    min_val = -(2 ** (int_bits + frac_bits))
    
    # 確保數值在允許的範圍內
    scaled_arr = np.clip(scaled_arr, min_val, max_val)
    
    # 將每個元素轉換為二進制
    def to_binary(val):
        if val < 0:
            val += 1 << (int_bits + frac_bits)
        return [int(x) for x in format(val, f'0{int_bits + frac_bits}b')]
    
    # 處理所有元素並生成二進制矩陣
    binary_matrix = np.array([to_binary(val) for val in scaled_arr.flatten()])
    
    # 重塑為原始形狀並增加額外的維度
    result_shape = arr.shape + (int_bits + frac_bits,)
    binary_matrix = binary_matrix.reshape(result_shape)
    
    return binary_matrix.astype(np.int8)

# 測試範例
input_array = np.array([[0.06983836, 0.02506763], [-0.04475388, 0.07006475]])
binary_result = float_to_fixed_point_binary(input_array)
print(binary_result)

print(type(binary_result[0][0][0]))