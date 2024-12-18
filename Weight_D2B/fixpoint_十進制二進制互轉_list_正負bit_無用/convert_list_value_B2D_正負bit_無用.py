# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:47:22 2024

@author: Otiz

用途: 把裝有二進制定點數的三維list轉成十進制二維list。
"""

def binary_fixed_point_to_decimal(binary_list, int_bits=2, frac_bits=2):

    sign = -1 if any(bit < 0 for bit in binary_list) else 1
    
    binary_list = [abs(bit) for bit in binary_list]
    
    int_part = binary_list[:int_bits]
    frac_part = binary_list[int_bits:int_bits + frac_bits]
    
    integer_value = 0
    for i, bit in enumerate(reversed(int_part)):
        if bit != 0:
            integer_value += (2 ** i)
    
    fractional_value = 0
    for i, bit in enumerate(frac_part, 1):
        if bit != 0:
            fractional_value += (2 ** -i)
    
    decimal_value = sign * (integer_value + fractional_value)
    return decimal_value

def convert_matrix_to_decimal(matrix, int_bits=2, frac_bits=2):
    result = []
    for row in matrix:
        new_row = []
        for binary_list in row:
            decimal_value = binary_fixed_point_to_decimal(binary_list, int_bits, frac_bits)
            new_row.append(decimal_value)
        result.append(new_row)
    return result

W = [[[1, 0, 1, 0], [0, -1, 0, -1]], 
     [[1, 0, 1, 0], [0, 1, 0, 1]]]
decimal_matrix = convert_matrix_to_decimal(W, int_bits=2, frac_bits=2)
print(decimal_matrix)

