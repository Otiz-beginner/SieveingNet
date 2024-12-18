# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:04:18 2024

@author: Otiz

用途: 把裝有十進制定點數的兩維list轉成二進制三維list。
"""

def decimal_to_binary_fixed_point(num):
    # 確定符號
    sign = -1 if num < 0 else 1
    num = abs(num)
    
    # 分離整數部分和小數部分
    integer_part = int(num)
    fractional_part = num - integer_part
    
    # 轉換整數部分
    int_bin = bin(integer_part)[2:]
    if len(int_bin) > 2:
        raise ValueError("Integer part exceeds two bits")
    
    # 轉換小數部分
    frac_bin = []
    while fractional_part and len(frac_bin) < 2:
        fractional_part *= 2
        bit = int(fractional_part)
        frac_bin.append(str(bit))
        fractional_part -= bit
    
    # 補齊二進制位數
    int_bin = int_bin.zfill(2)
    frac_bin = ''.join(frac_bin).ljust(2, '0')
    
    # 處理符號並轉換為列表
    binary_list = [int(x) * sign for x in int_bin + frac_bin]
    return binary_list

def convert_matrix_to_binary_fixed_point(matrix):
    result = []
    for row in matrix:
        new_row = []
        for num in row:
            binary_list = decimal_to_binary_fixed_point(num)
            new_row.append(binary_list)
        result.append(new_row)
    return result

# 測試範例
W = [[-2.5, 1.25], [2.5, 1.25]]
binary_matrix = convert_matrix_to_binary_fixed_point(W)
print(f"binary_matrix: {binary_matrix}")
print(f"type of binary_matrix: {type(binary_matrix[0][0][0])}")
