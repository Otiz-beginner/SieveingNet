# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 17:56:26 2024

@author: user
"""
import numpy as np

def decimal_to_binary_fixed_point_optimized(num, int_bits=1, frac_bits=15):
    # 确定数值的符号，负数设为 -1，正数设为 1
    sign = -1 if num < 0 else 1
    
    # 取数值的绝对值
    num = abs(num)
    
    # 分离整数部分和小数部分
    integer_part = int(num)
    fractional_part = num - integer_part
    
    # 将整数部分转换为二进制字符串，并保留指定的位数
    int_bin = f'{integer_part:0{int_bits}b}'[-int_bits:]
    
    # 使用 NumPy 优化小数部分转换
    factor = 2.0 ** np.arange(1, frac_bits + 1)
    print(factor)
    fractional_values = fractional_part * factor
    frac_bin_array = (fractional_values >= 1).astype(int)
    fractional_part -= np.sum(frac_bin_array * (1 / factor))
    
    # 将整数字符串和小数数组合并为 NumPy 数组
    int_bin_array = np.array([int(x) for x in int_bin], dtype=int)
    binary_array = np.concatenate((int_bin_array, frac_bin_array))
    
    # 如果原数值是负数，则将每一位数字转为负数
    if sign == -1:
        binary_array = -binary_array
    
    # 返回最终的二进制固定点表示 NumPy 数组
    return binary_array

def convert_matrix_to_binary_fixed_point_optimized(matrix, int_bits=1, frac_bits=15):
    # 将输入矩阵转换为 NumPy 数组
    matrix = np.array(matrix)
    shape = matrix.shape
    
    # 使用 NumPy 的 apply_along_axis 来批量转换矩阵中的每个元素
    flattened_matrix = matrix.flatten()
    flattened_result = np.array([decimal_to_binary_fixed_point_optimized(num, int_bits, frac_bits) for num in flattened_matrix])
    
    # 重塑结果为原始形状，增加一个维度来表示二进制位数
    reshaped_result = flattened_result.reshape(shape + (int_bits + frac_bits,))
    
    # 返回转换后的二进制固定点表示 NumPy 数组
    return reshaped_result


W = np.array([[0.5, -0.02506763], 
  [0.04475388, 0.07006475]])
binary_matrix = convert_matrix_to_binary_fixed_point_optimized(W, int_bits=1, frac_bits=15)
print("原始矩陣:")
print(W)
print("\n轉換後的矩陣:")
print(binary_matrix)
print(type(binary_matrix[0, 0, 0]))

# # 示例测试
# matrix = np.random.uniform(-5, 5, (3, 3))
# binary_matrix = convert_matrix_to_binary_fixed_point_optimized(matrix, int_bits=2, frac_bits=2)
# print("原始矩阵:")
# print(matrix)
# print("\n转换后的二进制固定点矩阵:")
# print(binary_matrix)




