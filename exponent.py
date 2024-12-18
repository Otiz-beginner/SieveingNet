# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:28:20 2024

@author: user
"""

import numpy as np

def exponent(x):
    if x.dtype == np.float32:
        packed_data = x.view(np.uint32)
        exponent_mask = 0x7F800000
        exponent_part_bias = (packed_data & exponent_mask) >> 23
        exponent_part = exponent_part_bias.view(np.int32) - 127
    elif x.dtype == np.float64:
        packed_data = x.view(np.uint64)
        exponent_mask = 0x7FF0000000000000
        exponent_part_bias = (packed_data & exponent_mask) >> 52
        exponent_part = exponent_part_bias.view(np.int64) - 1023
    else:
        raise ValueError("Only float32 and float64 are supported.")

    return exponent_part

# 测试
A_float32 = np.array([[-12.625, 4.75, 0], 
                      [12.625, 0.03125, 0]], dtype=np.float32)

# 測試浮點數矩陣的 exponent 部分提取
A_float32 = np.array([[12.625, 4.75, 0], 
                      [12.625, 0.03125, 0]], dtype=np.float32)

A_float64 = np.array([[12.625, 4.75, 0], 
                      [12.625, 0.03125, 0]], dtype=np.float64)

print("Float32 input:")
print(A_float32)
print("Float32 exponent parts:")
print(exponent(A_float32))

print("Float64 input:")
print(A_float64)
print("Float64 exponent parts:")
print(exponent(A_float64))
