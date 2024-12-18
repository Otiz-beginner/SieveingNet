# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 22:09:15 2024

@author: Otiz
"""
import numpy as np
def exponent(x):
    if x.dtype == np.float32:
        packed_data = x.view(np.uint32) # 先當作無號數看待，右移才會是unsigned extend數值才不會錯
        exponent_mask = 0x7F800000
        exponent_part_bias = (packed_data & exponent_mask) >> 23
        exponent_part = exponent_part_bias.view(np.int32) - 127 # 最後再看待乘有號數，才會有負數
    elif x.dtype == np.float64:
        packed_data = x.view(np.uint64)
        exponent_mask = 0x7FF0000000000000
        exponent_part_bias = (packed_data & exponent_mask) >> 52
        exponent_part = exponent_part_bias.view(np.int64) - 1023
    else:
        raise ValueError("Only float32 and float64 are supported.")
    return exponent_part

def Sieve_conventional(A, W, frac_bits, im):
    n = len(A)
    B = len(W[0])
    S = 0.0
    for k in range(B):
        S_kth = 0.0
        order = ((B-k)-1)
        power = order - frac_bits
        sivingThreshold = exponent(S) - im * (exponent(S) + 127) - power
        for i in range(n):
            if W[i][k] != 0:
                if exponent(A[i]) >= sivingThreshold:
                    S_kth += A[i]
        S += (S_kth * (2**power))
    return S

class SievingNet:
    def __init__(self):
        self.S_previous = 0.0
        
    def Sieve(self, A, W, frac_bits, im):
        n = len(A)
        B = len(W[0])
        S = 0.0
        for k in range(B):
            S_kth = 0.0
            order = ((B-k)-1)
            power = order - frac_bits
            sivingThreshold = exponent(self.S_previous) - im * (exponent(self.S_previous) + 127) - power
            for i in range(n):
                if W[i][k] != 0:
                    if exponent(A[i]) >= sivingThreshold:
                        S_kth += A[i]
            S += (S_kth * (2**power))
        return S

def float_to_fixed_point_binary(arr, int_bits=2, frac_bits=29):
    # 計算轉換係數
    scale = 2 ** frac_bits
    # 將浮點數乘以係數
    scaled_arr = np.round(arr * scale).astype(np.int64)
    # 計算整數的最大值和最小值
    max_val = (2 ** (int_bits + frac_bits)) - 1
    min_val = -(2 ** (int_bits + frac_bits))
    # 確保數值在允許的範圍內
    scaled_arr = np.clip(scaled_arr, min_val, max_val)
    def to_binary(val):
        if val >= 0:
            # 對於正數，直接轉成binary
            return np.unpackbits(np.array([val], dtype='>u8').view(np.uint8))[-(int_bits + frac_bits):]
            # >：Big-endian，u：Unsigned，2：two words（16bits），4: four words（32bits）
        else:
            # 對於正數，轉成binary最後再加上 - 號
            val = -val  # 先取絕對值
            bits = np.unpackbits(np.array([val], dtype='>u8').view(np.uint8))[-(int_bits + frac_bits):]
            return -bits
    # 將每個元素轉換為二進制
    binary_matrix = np.array([to_binary(val) for val in scaled_arr.flatten()])
    # 重塑為原始形狀並增加額外的維度
    result_shape = arr.shape + (int_bits + frac_bits,)
    binary_matrix = binary_matrix.reshape(result_shape)
    
    return binary_matrix.astype(np.int8)

def generate_random_matrices(num_samples, input_dim, output_dim, int_bits=4, frac_bits=28):
    # A matrix range [0, 10]  ReLU 活化後的節點值可能在 [0, 10] 範圍內，具體取決於網路深度和輸入資料分佈。
    A = np.random.uniform(0, 10, (num_samples, input_dim)).astype(np.float64)
    # W matrix range [-3, 3] 在神經網路的訓練過程中，權重值通常集中在較小的範圍內（例如 [-3, 3] 或 [-1, 1]）
    W = np.random.uniform(-3, 3, (input_dim, output_dim)).astype(np.float64)
    # Convert W matrix to binary form
    W_fixed_B = float_to_fixed_point_binary(W, int_bits, frac_bits)
    return A, W, W_fixed_B

np.random.seed(4)  # For reproducibility
num_samples = 1

input_dim = 4
output_dim = 1
int_bits = 4
frac_bits = 28
im = 1
# Generate random matrix
A, W_float, W_fixed_B = generate_random_matrices(num_samples, input_dim, output_dim, int_bits, frac_bits)

SievingNet = SievingNet()
print(Sieve_conventional(A, W_fixed_B, frac_bits, im))
print(SievingNet.Sieve(A, W_fixed_B, frac_bits, im))















