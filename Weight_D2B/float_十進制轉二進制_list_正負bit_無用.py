# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:08:43 2024

@author: Otis

用途: 裝有float的list十進制轉二進制
"""

def decimal_to_binary_fixed_point(num, int_bits=2, frac_bits=2):
    # 確定符號
    sign = -1 if num < 0 else 1
    num = abs(num)
    
    # 分離整數部分和小數部分
    integer_part = int(num)
    fractional_part = num - integer_part
    
    # 轉換整數部分
    int_bin = bin(integer_part)[2:]
    if len(int_bin) > int_bits:
        raise ValueError(f"Integer part exceeds {int_bits} bits")
    
    # 轉換小數部分
    frac_bin = []
    while fractional_part and len(frac_bin) < frac_bits:
        fractional_part *= 2
        bit = int(fractional_part)
        frac_bin.append(str(bit))
        fractional_part -= bit
    
    # 補齊二進制位數
    int_bin = int_bin.zfill(int_bits)
    frac_bin = ''.join(frac_bin).ljust(frac_bits, '0')
    
    # 處理符號並轉換為列表
    binary_list = [int(x) * sign for x in int_bin + frac_bin]
    return binary_list

def convert_matrix_to_binary_fixed_point(matrix, int_bits=2, frac_bits=2):
    result = []
    for row in matrix:
        new_row = []
        for num in row:
            binary_list = decimal_to_binary_fixed_point(num, int_bits, frac_bits)
            new_row.append(binary_list)
        result.append(new_row)
    return result

# 測試範例
# W = [[-2.5, 2.25], [12.625, 1.25]]
W = [[0.5, -0.02506763], 
 [0.04475388, 0.07006475]]
binary_matrix = convert_matrix_to_binary_fixed_point(W, int_bits=1, frac_bits=15)
print(binary_matrix)
print(type(binary_matrix[0][0][0]))

