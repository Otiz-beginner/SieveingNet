# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:37:02 2024

@author: user
"""

import numpy as np
import struct

def exponent(x):
    packed_data = struct.pack('f', x)
    binary_representation = struct.unpack('I', packed_data)[0]
    exponent_mask = 0xFF
    binary_representation_no_mantissa = binary_representation // 0x7FFFFF
    exponent_part = (binary_representation_no_mantissa & exponent_mask)
    return exponent_part - 127

def Sieve(A, W, B, W_p, p, ip, r):
    A = np.array(A, dtype=np.float32)
    W = np.array(W, dtype=np.int8)
    
    n_l = W.shape[1]
    batch = A.shape[0]
    X = np.zeros((batch, n_l), dtype=np.float)
    
    # 預計算
    powers = np.array([2 ** ((B - k - 1) - W_p) for k in range(B)], dtype=np.float32)
    W_nonzero_mask = W != 0
    
    for s in range(batch):
        for i in range(n_l):
            x = 0.0
            for k in range(B):
                order = (B - k) - 1
                sivingThreshold = r + ip * exponent(x) - order - p
                # 過濾非零權重並進行運算
                valid_indices = np.where(W_nonzero_mask[:, i, k])[0]
                if valid_indices.size > 0:
                    A_sub = A[s, valid_indices]
                    exponents = np.array([exponent(a) for a in A_sub], dtype=np.float32)
                    valid_indices = valid_indices[exponents >= sivingThreshold]
                    if valid_indices.size > 0:
                        x_kth = np.sum(A[s, valid_indices])
                        x += x_kth * powers[k]
            X[s, i] = x
    return X


# A = np.array([[12.625, 4.75], 
#               [12.625, 0.03125]], dtype=np.float32)

# W = np.array([[[1, 0, 1, 0], [0, 1, 0, 1]], 
#               [[1, 0, 1, 0], [0, 1, 0, 1]]], dtype=np.int8)

A = random_array = np.random.rand(1, 784)
W = np.random.uniform(0, 1e-1, (784, 512))

print(f"A = {A}")
print(f"correct answer: {np.dot(A, W)}")
print(f"correct answer type: {type(np.dot(A, W)[0, 0])}")
# print(f"before W = {W}")
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

W = float_to_fixed_point_binary(W)
# print(f"after W = {W}")


B = 16
W_p = 15
p = 10
ip = 0
r = 0

print("-------------------------------------------------------")
print(f"my answer: {Sieve(A, W, B, W_p, p, ip, r)}")
print(f"my answer type: {type(Sieve(A, W, B, W_p, p, ip, r)[0, 0])}")