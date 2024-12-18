# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:23:49 2024

@author: Otis

用途: 這一個是沒用的，幫出以為神經網路矩陣是三維結果只是二維，所以這支程式沒用。
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

def convert_list_to_binary_fixed_point(lst, int_bits=2, frac_bits=2):
    result = []
    for item in lst:
        if isinstance(item, list): # 如果還是list就在進行遞迴
            result.append(convert_list_to_binary_fixed_point(item, int_bits, frac_bits))
        else:
            result.append(decimal_to_binary_fixed_point(item, int_bits, frac_bits))
    return result

# 測試範例
W = [[[-2.5, 1.25], [2.5, 1.25]], [[-2.25], [2.25]]]
binary_matrix = convert_list_to_binary_fixed_point(W, int_bits=2, frac_bits=2)
print(binary_matrix)

