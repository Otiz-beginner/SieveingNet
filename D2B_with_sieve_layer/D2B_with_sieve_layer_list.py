# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 22:09:15 2024

@author: Otiz
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

import struct

def exponent(x):
    packed_data = struct.pack('f', x)
    binary_representation = struct.unpack('I', packed_data)[0]
    # 提取 exponent 部分
    exponent_mask = 0xFF
    binary_representation_no_mantissa = binary_representation // 0x7FFFFF
    exponent_part = binary_representation_no_mantissa & exponent_mask
    # 這一段是修正float=0的話沒有bias，但是因為我們的sieve當輸入為0就代表一定不用累加，
    # 所以現在float=0的bias=-127，會被我們過濾掉節省時間這正是我們要得，所以不修正。
    # exponent_part_bias = 0
    # if exponent_part != 0: # because float value is 0 the exponent will be no bias
    #     exponent_part_bias = exponent_part - 127 
    # return exponent_part_bias
    return exponent_part - 127



def Sieve(A, W, W_p, p, im, r):
    n_l = len(W) # last layer node
    n_n = len(W[0]) # next layer node
    B   = len(W[0][0]) # How many bits for a number
    samples = len(A) # how many samples at one
    print(f"n_l: {n_l}")
    print(f"n_n: {n_n}")
    print(f"B: {B}")
    print(f"samples: {samples}")
    print("-------------------------------------------------------")
    X = []
    for s in range(samples): # 這一層for可以做平行化
        x_sample = []
        print(f"****第{s+1}smaple")
        for j in range(n_n): # 這一層for可以做平行化
            x = 0.0
            print(f"***第{j+1}output x")
            for i in range(n_l): # 這一層使用for迴圈用stream實現
                x_item = 0.0
                print(f"**第{i+1}item in {j+1} output")
                for k in range(B): # 只有這一個for迴圈做成硬體
                    order = ((B-k)-1)
                    power = order - W_p
                    # 這一部分是設定output x範圍大小(因為輸入權重都有完全考慮)
                    sivingThreshold = r + im * exponent(x) - order - p
                    print(f"*第{k+1}bit of the {i+1} item in the {j+1} output")
                    # print("-------------------------------------------------------")
                    # print(f"r = {r}")
                    # print(f"im * exponent(x) = {im * exponent(x)}")
                    # print(f"order = {order}")
                    # print(f"power = {power}")
                    # print(f"sivingThreshold = {sivingThreshold}")
                    # print(f"exponent(A[s][i]) = {exponent(A[s][i])}")
                    # print("-------------------------------------------------------")
                    if W[i][j][k] != 0:
                        if exponent(A[s][i]) >= sivingThreshold:
                            x_item += (A[s][i] * (2**power))
                x += x_item
                print(f"第{s+1}個smaple的第{j+1}個output x = {x}")
                print()
            x_sample.append(x)
        X.append(x_sample)
    return X

A = [[12.625, 4.75, 0],  # type float normal case
    [12.625, 0.03125, 0]] # type float small case


W_o = [[2.5, 1.25], # W in decimal look likes
      [2.5, 1.25], 
      [2.5, 1.25]]
# W = [[[1, 0, 1, 0], [0, 1, 0, 1]],  
#      [[1, 0, 1, 0], [0, 1, 0, 1]], 
#      [[1, 0, 1, 0], [0, 1, 0, 1]] ]

input_array = np.array(W_o)
W = float_to_fixed_point_binary(input_array, int_bits=2, frac_bits=14)



# S = [[43.4375, 21.71875], 
#      [31.640625, 15.8203125]]
W_p = 14 # W_p is the precision of weight(the bits of after point)(in binary)
p = 5 # represent how many small of input can be(the bits of after point)(in decimal) 
im = 0 # im is the impact of exponent(S) when sieving
r = 0 
print("-------------------------------------------------------")
print(Sieve(A, W, W_p, p, im, r))

print(f"W: {W}")

