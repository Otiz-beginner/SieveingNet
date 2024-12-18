# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:31:41 2024

@author: user
"""
import numpy as np

import time

def float_to_fixed_point_binary(arr, int_bits=4, frac_bits=12):
    # 計算轉換係數
    scale = 2 ** frac_bits

    # 將浮點數乘以係數並四捨五入
    scaled_arr = np.round(arr * scale).astype(np.int32)

    # 計算整數的最大值和最小值
    max_val = (2 ** (int_bits + frac_bits)) - 1
    min_val = -(2 ** (int_bits + frac_bits))

    # 確保數值在允許的範圍內
    scaled_arr = np.clip(scaled_arr, min_val, max_val)
    
    # 处理正数和负数，负数直接取绝对值，并加上位数限制
    positive_scaled = scaled_arr >= 0
    negative_scaled = scaled_arr < 0
    
    # 创建存储二进制结果的数组
    result_shape = arr.shape + (int_bits + frac_bits,)
    binary_matrix = np.zeros(result_shape, dtype=np.int8)

    # 处理正数
    positive_values = scaled_arr[positive_scaled]
    if positive_values.size > 0:
        positive_bits = np.unpackbits(positive_values.astype(np.uint16).view(np.uint8)).reshape(-1, 16)[:, -result_shape[-1]:]
        binary_matrix[positive_scaled] = positive_bits

    # 处理负数
    negative_values = -scaled_arr[negative_scaled]
    if negative_values.size > 0:
        negative_bits = np.unpackbits(negative_values.astype(np.uint16).view(np.uint8)).reshape(-1, 16)[:, -result_shape[-1]:]
        binary_matrix[negative_scaled] = -negative_bits

    return binary_matrix


# 測試範例
input_array = np.array([[12.625, 0.02506763], [-0.04475388, 0.07006475]])
start_time = time.time()
for t in range(100000):
    binary_result = float_to_fixed_point_binary(input_array, int_bits=4, frac_bits=12)
end_time = time.time()

print(binary_result)

print(f"total time: {end_time - start_time}")