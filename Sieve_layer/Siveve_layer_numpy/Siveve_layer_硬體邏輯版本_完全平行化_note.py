# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:25:58 2024

@author: user
"""

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

def Sieve(A, W, W_p, p, im, r):
    A = np.array(A, dtype=np.float32)
    W = np.array(W, dtype=np.int8)
    W_p = np.int32(W_p)
    p = np.int32(p)
    im = np.float32(im)
    r = np.int32(r)

    n_l, n_n, B = W.shape
    samples = A.shape[0]
    X = np.zeros((samples, n_n), dtype=np.float32)

    bit_index = np.arange(B - 1, -1, -1, dtype=np.float32)
    power = bit_index - W_p  # power.shape = (B,)

    # Expand bit index and power for broadcasting
    bit_index_expanded = bit_index[np.newaxis, np.newaxis, :]  # shape (1, 1, B)
    power_expanded = power[np.newaxis, np.newaxis, :]  # shape (1, 1, B)

    # Loop over each input layer node
    for i in range(n_l):
        # Compute exponents for input A
        exponents_A = exponent(A[:, i])[:, np.newaxis]  # exponents_A.shape = (samples, 1)

        # Expand A for broadcasting
        A_expanded = A[:, i, np.newaxis, np.newaxis]  # shape (samples, 1, 1)

        # Compute sivingThreshold based on X
        exponents_X = exponent(X)[:, :, np.newaxis]  # exponents_X.shape = (samples, n_n, 1)
        sivingThreshold = r + im * exponents_X - bit_index_expanded - p  # shape (samples, n_n, B)

        # Create mask where W is non-zero
        mask = W[i, :, :] # mask.shape = (n_n, B)

        # Calculate contribution for each bit
        contribution = np.where(
            exponents_A[:, np.newaxis, :] >= sivingThreshold,  # Compare A exponents with sivingThreshold
            A_expanded * (2.0**power_expanded),  # shape (samples, 1, B)
            0.0
        )  # contribution.shape = (samples, n_n, B)

        # Sum contributions across the bits and apply mask
        X += np.sum(mask * contribution, axis=-1)  # Sum over the bit axis, shape (samples, n_n)
    
    return X

# 測試函數
A = np.array([[12.625, 4.75, 0], 
    [12.625, 0.03125, 0]])

W = np.array([[-2.5, 1.25], 
      [-2.5, 1.25], 
      [-2.5, 1.25]])

S = np.dot(A, W)
print(f"S: {S}")

W_p = np.int32(14)
p = np.int32(5)
im = np.float32(0.0)
r = np.int32(0)
W = float_to_fixed_point_binary(W, int_bits=2, frac_bits=W_p)

print(Sieve(A, W, W_p, p, im, r))
